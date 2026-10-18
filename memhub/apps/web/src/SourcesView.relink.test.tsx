// @vitest-environment jsdom
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { RecallHostStatus, RecallPermissionStatus } from "./global";
import { SourcesView } from "./SourcesView";
import type { useImportSources } from "./use-import-sources";

afterEach(() => {
  cleanup();
  delete window.oneMemory;
});

function setup(
  reason: RecallPermissionStatus["reason"] = "legacy_installation",
) {
  let permission: RecallPermissionStatus = {
    hostId: "cursor",
    state: "relink-required",
    reason,
    receipt: null,
  };
  const installedPermission: RecallPermissionStatus = {
    hostId: "cursor",
    state: "install-only",
    reason: null,
    receipt: {
      hostId: "cursor",
      installationId: "new-installation",
      revision: 4,
      status: "install-only",
      allowedTools: [],
      allowedScopes: [],
    },
  };
  const host: RecallHostStatus = {
    hostId: "cursor",
    label: "Cursor",
    present: true,
    installed: true,
    selfCheck: {
      state: "wired",
      probed_at: "2026-09-15T01:00:00Z",
      detail: "old profile",
    },
    skillDrift: { state: "in_sync", installedVersion: 9, currentVersion: 9 },
    permission,
  };
  let installationCompleted = false;
  const api = {
    getApiBaseUrl: vi.fn(),
    getApiToken: vi.fn(),
    openHandoff: vi.fn(),
    getRecallHosts: vi.fn().mockImplementation(async () => [
      {
        ...host,
        permission,
        selfCheck: installationCompleted ? null : host.selfCheck,
      },
    ]),
    previewRecallInstall: vi.fn().mockResolvedValue({
      hostId: "cursor",
      hostLabel: "Cursor",
      fingerprint: "current-profile-fingerprint",
      files: [
        {
          path: "/synthetic/.cursor/mcp.json",
          owner: "host",
          existed: true,
          hunk: {
            startLine: 1,
            removed: ["old-profile"],
            added: ["current-profile"],
            contextBefore: [],
            contextAfter: [],
          },
        },
      ],
    }),
    installRecall: vi.fn().mockImplementation(async () => {
      installationCompleted = true;
      permission = installedPermission;
      // The authoritative GET, not this older operation receipt, supplies grant CAS.
      return {
        hostId: "cursor",
        installed: true,
        permission: {
          ...installedPermission,
          receipt: {
            ...installedPermission.receipt,
            installationId: "install-response-identity",
            revision: 1,
          },
        },
      };
    }),
    previewRecallUninstall: vi.fn().mockResolvedValue({
      hostId: "cursor",
      hostLabel: "Cursor",
      fingerprint: "remove",
      files: [],
    }),
    uninstallRecall: vi.fn(),
    getRecallPermission: vi.fn().mockImplementation(async () => permission),
    setRecallPermission: vi.fn().mockImplementation(async (hostId, input) => {
      permission = {
        hostId,
        state: "approved",
        reason: null,
        receipt: {
          hostId,
          installationId: input.installationId,
          revision: input.expectedRevision + 1,
          status: "approved",
          allowedTools: input.allowedTools,
          allowedScopes: input.allowedScopes,
        },
      };
      return permission;
    }),
    revokeRecallPermission: vi.fn(),
    probeRecallHost: vi.fn(),
    previewSkillSync: vi.fn(),
    applySkillSync: vi.fn(),
  };
  window.oneMemory = api;
  const imports = {
    sources: [
      {
        source_id: "cursor",
        display_name: "Cursor",
        authorized: true,
        available: true,
        root_id: "synthetic-root",
        grant_revision: 1,
        imported_messages: 1,
        imported_conversations: 1,
        candidate_path: "/synthetic/cursor",
        candidate_kind: "default",
        storage_kinds: ["jsonl"],
        sync_completeness: "additions_only",
        capabilities: {
          historical_import: "implemented",
          incremental_import: "implemented",
          live_sync: "not_implemented",
          recall: "not_implemented",
          installer: "not_implemented",
          verification: "fixture_only",
        },
      },
    ],
    jobs: [],
    latestJobBySource: new Map(),
    loading: false,
    busyKey: null,
    error: null,
    lastConnectResult: null,
    refresh: vi.fn(),
    connectSelected: vi.fn(),
    syncAll: vi.fn(),
    startImport: vi.fn(),
    changeJob: vi.fn(),
  } as unknown as ReturnType<typeof useImportSources>;
  render(<SourcesView imports={imports} />);
  return api;
}

describe("existing connection belongs to a different profile", () => {
  it.each([
    "legacy_installation",
    "credential_missing_or_changed",
    "managed_config_changed",
  ] as const)(
    "offers a reviewed reconnect instead of stale wired status: %s",
    async (reason) => {
      const api = setup(reason);
      const user = userEvent.setup();
      const button = await screen.findByRole("button", {
        name: "Reconnect in one click",
      });
      expect(screen.queryByText("Link: Connected")).toBeNull();
      expect(screen.getByText("Link: needs reconnection")).toBeTruthy();
      expect(
        screen.getByRole("button", { name: "Allow memory use" }),
      ).toHaveProperty("disabled", true);
      await user.click(button);
      expect(api.previewRecallInstall).toHaveBeenCalledWith("cursor");
      expect(api.previewRecallUninstall).not.toHaveBeenCalled();
      expect(api.installRecall).not.toHaveBeenCalled();
      expect(api.setRecallPermission).not.toHaveBeenCalled();
      expect((await screen.findByRole("dialog")).textContent).toContain(
        "current-profile",
      );
      await user.click(screen.getByRole("button", { name: "Cancel" }));
      expect(api.installRecall).not.toHaveBeenCalled();
    },
  );

  it("reconnects only after explicit combined consent and grants the newly read identity", async () => {
    const api = setup();
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Reconnect in one click" }),
    );
    expect(screen.getByRole("dialog").textContent).toContain(
      "Confirming also allows Cursor",
    );
    expect(api.installRecall).not.toHaveBeenCalled();
    expect(api.setRecallPermission).not.toHaveBeenCalled();
    await user.click(
      screen.getByRole("button", { name: "Connect and allow memory use" }),
    );
    expect(api.installRecall).toHaveBeenCalledTimes(1);
    expect(api.installRecall).toHaveBeenCalledWith(
      "cursor",
      "current-profile-fingerprint",
    );
    await waitFor(() =>
      expect(api.setRecallPermission).toHaveBeenCalledExactlyOnceWith(
        "cursor",
        {
          installationId: "new-installation",
          expectedRevision: 4,
          allowedTools: [
            "om_recall",
            "om_ask",
            "om_conversations",
            "om_diagnostic_trace",
          ],
          allowedScopes: ["all-apps", "source:cursor"],
          userConfirmed: true,
        },
      ),
    );
    expect(api.getRecallPermission).toHaveBeenCalledExactlyOnceWith("cursor");
    expect(api.installRecall.mock.invocationCallOrder[0]).toBeLessThan(
      api.getRecallPermission.mock.invocationCallOrder[0],
    );
    expect(api.getRecallPermission.mock.invocationCallOrder[0]).toBeLessThan(
      api.setRecallPermission.mock.invocationCallOrder[0],
    );
    await waitFor(() => expect(api.getRecallHosts).toHaveBeenCalledTimes(2));
    expect(
      await screen.findByRole("button", { name: "Adjust call permissions" }),
    ).toBeTruthy();
    expect(screen.queryByText("Link: Connected")).toBeNull();
    expect(api.uninstallRecall).not.toHaveBeenCalled();
  });

  it("keeps the reconnect action available after a preview failure without changing configuration", async () => {
    const api = setup();
    api.previewRecallInstall.mockRejectedValueOnce(
      new Error("controlled preview failure"),
    );
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Reconnect in one click" }),
    );
    expect((await screen.findByRole("alert")).textContent).toContain(
      "Could not read the connection change preview",
    );
    expect(
      screen.getByRole("button", { name: "Reconnect in one click" }),
    ).toHaveProperty("disabled", false);
    expect(
      screen.queryByText("Cannot read the installation state."),
    ).toBeNull();
    expect(api.installRecall).not.toHaveBeenCalled();
    await user.click(
      screen.getByRole("button", { name: "Reconnect in one click" }),
    );
    expect(await screen.findByRole("dialog")).toBeTruthy();
  });

  it("keeps installation but never claims approval when the confirmed grant fails", async () => {
    const api = setup();
    api.setRecallPermission.mockRejectedValueOnce(
      new Error("controlled authorization failure"),
    );
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Reconnect in one click" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Connect and allow memory use" }),
    );
    await waitFor(() => expect(api.getRecallHosts).toHaveBeenCalledTimes(2));
    expect(api.installRecall).toHaveBeenCalledExactlyOnceWith(
      "cursor",
      "current-profile-fingerprint",
    );
    expect(api.setRecallPermission).toHaveBeenCalledExactlyOnceWith("cursor", {
      installationId: "new-installation",
      expectedRevision: 4,
      allowedTools: [
        "om_recall",
        "om_ask",
        "om_conversations",
        "om_diagnostic_trace",
      ],
      allowedScopes: ["all-apps", "source:cursor"],
      userConfirmed: true,
    });
    expect(
      await screen.findByText("Installed; memory use is not allowed yet"),
    ).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Allow memory use" }),
    ).toHaveProperty("disabled", false);
    expect(
      screen.queryByRole("button", { name: "Adjust call permissions" }),
    ).toBeNull();
    expect(api.uninstallRecall).not.toHaveBeenCalled();
  });

  it("does not authorize a stale relink-required identity returned after installation", async () => {
    const api = setup();
    api.getRecallPermission.mockResolvedValue({
      hostId: "cursor",
      state: "relink-required",
      reason: "managed_config_changed",
      receipt: null,
    });
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Reconnect in one click" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Connect and allow memory use" }),
    );
    await waitFor(() => expect(api.getRecallHosts).toHaveBeenCalledTimes(2));
    expect(api.getRecallPermission).toHaveBeenCalledExactlyOnceWith("cursor");
    expect(api.setRecallPermission).not.toHaveBeenCalled();
    expect(api.installRecall).toHaveBeenCalledTimes(1);
  });
});
