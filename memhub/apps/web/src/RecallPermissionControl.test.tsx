// @vitest-environment jsdom

import type { SourceResponse } from "@one-memory/contracts";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type {
  RecallHostStatus,
  RecallPermissionReceipt,
  RecallPermissionStatus,
  RecallPermissionTool,
} from "./global";
import {
  RecallPermissionControl,
  type RecallPermissionControlProps,
} from "./RecallPermissionControl";

afterEach(() => {
  cleanup();
  delete window.oneMemory;
});

function baseReceipt(): RecallPermissionReceipt {
  return {
    hostId: "workbuddy",
    installationId: "isolated-installation-a",
    revision: 1,
    status: "install-only",
    allowedTools: [],
    allowedScopes: [],
  };
}

function permission(
  overrides: Partial<RecallPermissionStatus> = {},
): RecallPermissionStatus {
  return {
    hostId: "workbuddy",
    state: "install-only",
    reason: null,
    receipt: baseReceipt(),
    ...overrides,
  };
}

function approved(
  tools: readonly RecallPermissionTool[] = ["om_recall"],
  scopes: readonly string[] = ["all-apps"],
): RecallPermissionStatus {
  const base = permission();
  return {
    ...base,
    state: "approved",
    receipt: {
      ...baseReceipt(),
      revision: 2,
      status: "approved",
      allowedTools: tools,
      allowedScopes: scopes,
    },
  };
}

function host(
  status: RecallPermissionStatus | undefined = permission(),
): RecallHostStatus {
  return {
    hostId: "workbuddy",
    label: "WorkBuddy",
    present: true,
    installed: true,
    selfCheck: null,
    skillDrift: null,
    permission: status,
  };
}

function source(id = "workbuddy", authorized = true): SourceResponse {
  return {
    source_id: id,
    display_name: id === "workbuddy" ? "WorkBuddy" : "Cursor",
    authorized,
    available: true,
    detected: true,
    storage_kinds: ["jsonl"],
    root_label: `/isolated/${id}`,
    candidate_path: `/isolated/${id}`,
    candidate_kind: "default",
    root_id: `isolated-${id}`,
    grant_revision: 1,
    imported_messages: 2,
    imported_conversations: 1,
    sync_enabled: false,
    sync_completeness: "additions_only",
    capabilities: {
      historical_import: "implemented",
      incremental_import: "implemented",
      live_sync: "implemented",
      recall: "not_implemented",
      installer: "not_implemented",
      verification: "fixture_only",
    },
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

function setup(overrides: Partial<RecallPermissionControlProps> = {}) {
  let current = overrides.host?.permission ?? permission();
  const get = vi.fn().mockImplementation(async () => current);
  const set = vi.fn().mockImplementation(async (_id, input) => {
    current = {
      hostId: "workbuddy",
      state: "approved",
      reason: null,
      receipt: {
        hostId: "workbuddy",
        installationId: input.installationId,
        revision: input.expectedRevision + 1,
        status: "approved",
        allowedTools: [...input.allowedTools],
        allowedScopes: [...input.allowedScopes],
      },
    };
    return current;
  });
  const revoke = vi.fn().mockImplementation(async (_id, input) => {
    current = {
      hostId: "workbuddy",
      state: "revoked",
      reason: null,
      receipt: {
        hostId: "workbuddy",
        installationId: input.installationId,
        revision: input.expectedRevision + 1,
        status: "revoked",
        allowedTools: [],
        allowedScopes: [],
      },
    };
    return current;
  });
  const install = vi.fn();
  const query = vi.fn();
  const changed = vi.fn();
  window.oneMemory = {
    getRecallPermission: get,
    setRecallPermission: set,
    revokeRecallPermission: revoke,
    installRecall: install,
    probeRecallHost: query,
  } as unknown as NonNullable<Window["oneMemory"]>;
  let props: RecallPermissionControlProps = {
    host: host(),
    sources: [source(), source("cursor")],
    onChanged: changed,
    ...overrides,
  };
  const view = render(<RecallPermissionControl {...props} />);
  return {
    ...view,
    get,
    set,
    revoke,
    install,
    query,
    changed,
    update(next: Partial<RecallPermissionControlProps>) {
      props = { ...props, ...next };
      view.rerender(<RecallPermissionControl {...props} />);
    },
  };
}

const button = (name: string) =>
  screen.getByRole<HTMLButtonElement>("button", { name });
const check = (name: string) =>
  screen.getByRole<HTMLInputElement>("checkbox", { name });
/** The sheet's own commit, whatever the sheet is editing. */
const commitButton = () =>
  screen.getByRole<HTMLButtonElement>("button", {
    name: /Allow and save|Save this scope/,
  });

/** Open the scope sheet. The row's "allow" grants without opening it. */
async function open() {
  await userEvent.click(button("Adjust scope"));
}
async function confirm() {
  await userEvent.click(commitButton());
}
const defaultTools: RecallPermissionTool[] = [
  "om_recall",
  "om_ask",
  "om_conversations",
  "om_diagnostic_trace",
];
const defaultScopes = ["all-apps", "source:cursor", "source:workbuddy"];
/** The receipt the mock echoes back for a full default grant. */
const approvedAll = () => approved(defaultTools, defaultScopes);

describe("RecallPermissionControl", () => {
  it("allows everything in one click from the row, after rechecking the installation identity", async () => {
    const api = setup();
    expect(
      screen.getByText("Installed; memory use is not allowed yet"),
    ).toBeTruthy();
    await userEvent.click(button("Allow memory use"));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(api.get).toHaveBeenCalledExactlyOnceWith("workbuddy");
    expect(api.set).toHaveBeenCalledExactlyOnceWith("workbuddy", {
      installationId: "isolated-installation-a",
      expectedRevision: 1,
      allowedTools: defaultTools,
      allowedScopes: defaultScopes,
      userConfirmed: true,
    });
    expect(api.changed).toHaveBeenCalledTimes(1);
    expect(api.install).not.toHaveBeenCalled();
    expect(api.query).not.toHaveBeenCalled();
    expect(screen.getByText("Allowed WorkBuddy to use memory.")).toBeTruthy();
    expect(
      screen.getByText("Memory use is allowed for the selected scopes"),
    ).toBeTruthy();
  });

  it("does not spend a row allow on a connection that changed underneath it", async () => {
    const api = setup();
    api.get.mockResolvedValue(
      permission({ receipt: { ...baseReceipt(), revision: 3 } }),
    );
    await userEvent.click(button("Allow memory use"));
    expect(api.set).not.toHaveBeenCalled();
    expect(screen.getByRole("alert").textContent).toContain("Close and reopen");
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.queryByText("Allowed WorkBuddy to use memory.")).toBeNull();
  });

  it("keeps a row allow that succeeded even if the parent refresh fails", async () => {
    const api = setup({
      onChanged: vi.fn().mockRejectedValue(new Error("parent refresh failed")),
    });
    await userEvent.click(button("Allow memory use"));
    expect(api.set).toHaveBeenCalledTimes(1);
    expect(screen.getByText("Allowed WorkBuddy to use memory.")).toBeTruthy();
    expect(screen.getByRole("alert").textContent).toContain(
      "The authorization was saved, but the list refresh failed",
    );
  });

  it("keeps a failed row allow inline, without repeating it", async () => {
    const api = setup();
    api.set.mockRejectedValueOnce(new Error("unknown private diagnostic"));
    await userEvent.click(button("Allow memory use"));
    expect(screen.getByRole("alert").textContent).toContain(
      "This grant could not be confirmed",
    );
    expect(screen.queryByText(/private diagnostic/)).toBeNull();
    expect(button("Allow memory use").disabled).toBe(false);
    await userEvent.click(button("Allow memory use"));
    expect(api.set).toHaveBeenCalledTimes(2);
    expect(api.set.mock.calls[1][1]).toEqual(api.set.mock.calls[0][1]);
  });

  it("opens the sheet with every offered tool and scope already selected, and writes nothing until it is submitted", async () => {
    const api = setup();
    await open();
    expect(check("Retrieve memories").checked).toBe(true);
    expect(check("Memory Q&A (optional)").checked).toBe(true);
    expect(check("Read raw conversations").checked).toBe(true);
    expect(check("Cross-app shared memory (all-apps)").checked).toBe(true);
    expect(check("WorkBuddy raw conversations").checked).toBe(true);
    expect(check("Cursor raw conversations").checked).toBe(true);
    for (const fn of [
      api.get,
      api.set,
      api.revoke,
      api.install,
      api.query,
      api.changed,
    ])
      expect(fn).not.toHaveBeenCalled();
    await confirm();
    expect(api.get).toHaveBeenCalledExactlyOnceWith("workbuddy");
    expect(api.set).toHaveBeenCalledExactlyOnceWith("workbuddy", {
      installationId: "isolated-installation-a",
      expectedRevision: 1,
      allowedTools: defaultTools,
      allowedScopes: defaultScopes,
      userConfirmed: true,
    });
    expect(api.changed).toHaveBeenCalledTimes(1);
    expect(api.install).not.toHaveBeenCalled();
    expect(api.query).not.toHaveBeenCalled();
    expect(
      screen.getByText(
        /Authorization does not mean the app has actually called or adopted memories/,
      ),
    ).toBeTruthy();
  });

  it.each(["Cancel", "Close", "escape"])(
    "%s restores focus with no IPC calls and restores the default selection",
    async (action) => {
      const api = setup();
      const trigger = button("Adjust scope");
      await open();
      await userEvent.click(check("Retrieve memories"));
      if (action === "escape") await userEvent.keyboard("{Escape}");
      else await userEvent.click(button(action));
      expect(screen.queryByRole("dialog")).toBeNull();
      expect(document.activeElement).toBe(trigger);
      expect(api.get).not.toHaveBeenCalled();
      expect(api.set).not.toHaveBeenCalled();
      await open();
      expect(check("Retrieve memories").checked).toBe(true);
    },
  );

  it("disables the allow button when a narrowed selection is no longer grantable", async () => {
    const api = setup();
    await open();
    for (const name of [
      "Retrieve memories",
      "Memory Q&A (optional)",
      "Read raw conversations",
      "Read redacted diagnostic records",
    ])
      await userEvent.click(check(name));
    expect(button("Allow and save").disabled).toBe(true);
    await userEvent.click(check("Retrieve memories"));
    expect(button("Allow and save").disabled).toBe(false);
    await userEvent.click(check("Cross-app shared memory (all-apps)"));
    expect(button("Allow and save").disabled).toBe(true);
    expect(api.get).not.toHaveBeenCalled();
    expect(api.set).not.toHaveBeenCalled();
  });

  it("offers only authorized, unambiguous sources, and sends the default selection unchanged", async () => {
    const api = setup({ sources: [source(), source("cursor", false)] });
    await open();
    expect(
      screen.queryByRole("checkbox", { name: "Cursor raw conversations" }),
    ).toBeNull();
    expect(check("WorkBuddy raw conversations").checked).toBe(true);
    await confirm();
    expect(api.set.mock.calls[0][1].allowedTools).toEqual(defaultTools);
    expect(api.set.mock.calls[0][1].allowedScopes).toEqual([
      "all-apps",
      "source:workbuddy",
    ]);
  });

  it("discloses that ask may use the configured model, and lets it be left out", async () => {
    const api = setup();
    await open();
    expect(screen.getByText(/May use the configured model/)).toBeTruthy();
    await userEvent.click(check("Memory Q&A (optional)"));
    await confirm();
    expect(api.set.mock.calls[0][1].allowedTools).toEqual([
      "om_recall",
      "om_conversations",
      "om_diagnostic_trace",
    ]);
  });

  it("offers only explicitly supplied valid workspace scopes, and includes them by default", async () => {
    const scope = `workspace:v1:${"a".repeat(64)}`;
    const api = setup({
      workspaceScopes: [
        { scope, label: "Isolated project" },
        { scope: "/not-a-scope", label: "Unavailable path" },
      ],
    });
    await open();
    expect(
      screen.queryByRole("checkbox", { name: "Unavailable path" }),
    ).toBeNull();
    expect(check("Workspace: Isolated project").checked).toBe(true);
    await confirm();
    expect(api.set.mock.calls[0][1].allowedScopes).toEqual([
      ...defaultScopes,
      scope,
    ]);
  });

  it("initializes edits from the receipt, discloses replacement and removals, and replaces on submit", async () => {
    const initial = approved(["om_recall"], ["all-apps", "source:workbuddy"]);
    const api = setup({ host: host(initial) });
    await userEvent.click(button("Adjust call permissions"));
    expect(check("Retrieve memories").checked).toBe(true);
    expect(check("Cross-app shared memory (all-apps)").checked).toBe(true);
    expect(check("WorkBuddy raw conversations").checked).toBe(true);
    expect(
      screen.getByText(/replaces all current tools and scopes/),
    ).toBeTruthy();
    await userEvent.click(check("WorkBuddy raw conversations"));
    expect(screen.getByText(/This grant removes/).textContent).toContain(
      "source:workbuddy",
    );
    await confirm();
    expect(api.set.mock.calls[0][1].expectedRevision).toBe(2);
    expect(api.set.mock.calls[0][1].allowedScopes).toEqual(["all-apps"]);
  });

  it("does not silently lose an existing workspace or missing source scope that cannot be verified", async () => {
    const unknown = `workspace:v1:${"b".repeat(64)}`;
    const api = setup({
      host: host(
        approved(["om_recall"], ["all-apps", unknown, "source:absent"]),
      ),
    });
    await userEvent.click(button("Adjust call permissions"));
    expect(button("Save this scope").disabled).toBe(true);
    expect(check(`Remove unverified scope: ${unknown}`).checked).toBe(false);
    expect(check("Remove unverified scope: source:absent").checked).toBe(false);
    await userEvent.click(check(`Remove unverified scope: ${unknown}`));
    await userEvent.click(check("Remove unverified scope: source:absent"));
    await confirm();
    expect(api.set.mock.calls[0][1].allowedScopes).toEqual(["all-apps"]);
  });

  it.each(["revision", "installation", "relink"])(
    "blocks stale %s before writes and requires reopening",
    async (change) => {
      const api = setup();
      await open();
      const next = permission();
      if (change === "revision")
        next.receipt = { ...baseReceipt(), revision: 3 };
      if (change === "installation")
        next.receipt = {
          ...baseReceipt(),
          installationId: "isolated-installation-b",
        };
      if (change === "relink") next.state = "relink-required";
      api.get.mockResolvedValue(next);
      await confirm();
      expect(api.set).not.toHaveBeenCalled();
      expect(screen.getByRole("alert").textContent).toContain(
        "Close and reopen",
      );
      expect(button("Allow and save").disabled).toBe(true);
    },
  );

  it("latches changed props even if a later poll returns the old receipt", async () => {
    const api = setup();
    await open();
    api.update({ host: host(approvedAll()) });
    api.update({ host: host() });
    expect(screen.getByRole("alert").textContent).toContain("Close and reopen");
    expect(button("Allow and save").disabled).toBe(true);
    expect(api.set).not.toHaveBeenCalled();
  });

  it("freezes receipt arrays and revision instead of sharing mutable polling references", async () => {
    const original = approved();
    const api = setup({ host: host(original) });
    await userEvent.click(button("Adjust call permissions"));
    if (!original.receipt) throw new Error("test receipt missing");
    (original.receipt.allowedScopes as string[]).push("source:cursor");
    api.get.mockResolvedValue(approved());
    await confirm();
    expect(api.set.mock.calls[0][1].allowedScopes).toEqual(["all-apps"]);
    expect(api.set.mock.calls[0][1].expectedRevision).toBe(2);
  });

  it.each(["missing", "revoked", "root"])(
    "blocks a selected raw source becoming %s",
    async (change) => {
      const api = setup();
      await open();
      const next = source();
      if (change === "revoked") next.authorized = false;
      if (change === "root") next.root_id = "replaced";
      api.update({ sources: change === "missing" ? [] : [next] });
      expect(screen.getByRole("alert").textContent).toContain(
        "Close and reopen",
      );
      expect(button("Allow and save").disabled).toBe(true);
      expect(api.set).not.toHaveBeenCalled();
    },
  );

  it("ignores late metadata after cancellation and a newer modal has opened", async () => {
    const api = setup();
    const late = deferred<RecallPermissionStatus>();
    api.get.mockReturnValueOnce(late.promise);
    await open();
    await confirm();
    await userEvent.click(button("Cancel"));
    await open();
    await act(async () => late.resolve(approved()));
    expect(check("Retrieve memories").checked).toBe(true);
    expect(api.set).not.toHaveBeenCalled();
    expect(api.changed).not.toHaveBeenCalled();
  });

  it("ignores a read completed after unmount", async () => {
    const api = setup();
    const late = deferred<RecallPermissionStatus>();
    api.get.mockReturnValueOnce(late.promise);
    await open();
    await confirm();
    api.unmount();
    await act(async () => late.resolve(permission()));
    expect(api.set).not.toHaveBeenCalled();
    expect(api.changed).not.toHaveBeenCalled();
  });

  it("keeps one pending mutation and prevents closing or resubmitting during commit", async () => {
    const api = setup();
    const write = deferred<RecallPermissionStatus>();
    api.set.mockReturnValueOnce(write.promise);
    await open();
    const submit = button("Allow and save");
    act(() => {
      fireEvent.click(submit);
      fireEvent.click(submit);
    });
    await act(async () => {});
    expect(api.get).toHaveBeenCalledTimes(1);
    expect(api.set).toHaveBeenCalledTimes(1);
    expect(button("Cancel").disabled).toBe(true);
    expect(button("Close").disabled).toBe(true);
    await userEvent.keyboard("{Escape}");
    expect(screen.getByRole("dialog").getAttribute("aria-busy")).toBe("true");
    await act(async () => write.resolve(approvedAll()));
    expect(screen.getByText("The call authorization was saved.")).toBeTruthy();
  });

  it("keeps errors inline, refreshes before retry, and never reuses consent", async () => {
    const api = setup();
    api.set.mockRejectedValueOnce(new Error("unknown private diagnostic"));
    await open();
    await confirm();
    expect(screen.getByRole("alert").textContent).toContain(
      "The save result is not confirmed yet",
    );
    expect(screen.queryByText(/private diagnostic/)).toBeNull();
    expect(button("Allow and save").disabled).toBe(true);
    await userEvent.click(button("Refresh and check"));
    expect(api.set).toHaveBeenCalledTimes(1);
    expect(button("Allow and save").disabled).toBe(false);
    await confirm();
    expect(api.set).toHaveBeenCalledTimes(2);
    expect(api.set.mock.calls[1][1]).toEqual(api.set.mock.calls[0][1]);
  });

  it("does not retry when refresh fails, or when an uncertain write already advanced the receipt", async () => {
    const api = setup();
    api.set.mockRejectedValueOnce(new Error("timeout"));
    await open();
    await confirm();
    api.get.mockRejectedValueOnce(new Error("offline"));
    await userEvent.click(button("Refresh and check"));
    expect(api.set).toHaveBeenCalledTimes(1);
    expect(button("Allow and save").disabled).toBe(true);
    api.get.mockResolvedValue(approved());
    await userEvent.click(button("Refresh and check"));
    expect(api.set).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("alert").textContent).toContain("Close and reopen");
  });

  it("keeps a committed receipt if parent refresh fails, offering read-only retry", async () => {
    const api = setup({
      onChanged: vi.fn().mockRejectedValue(new Error("parent refresh failed")),
    });
    await open();
    await confirm();
    expect(screen.getByRole("alert").textContent).toContain(
      "The authorization was saved, but the list refresh failed",
    );
    expect(screen.queryByRole("button", { name: "Allow and save" })).toBeNull();
    await userEvent.click(button("Refresh and check"));
    expect(api.set).toHaveBeenCalledTimes(1);
    expect(screen.getByText("The call authorization was saved.")).toBeTruthy();
  });

  it("does not treat an incomplete or mismatched write response as confirmed success", async () => {
    const api = setup();
    api.set.mockResolvedValueOnce(permission({ receipt: null }));
    await open();
    await confirm();
    expect(screen.getByRole("alert").textContent).toContain(
      "The save result is not confirmed yet",
    );
    expect(screen.queryByText("The call authorization was saved.")).toBeNull();
    expect(api.changed).not.toHaveBeenCalled();
  });

  it("revokes only the frozen installation after separate confirmation, without uninstalling or deleting sources", async () => {
    const api = setup({ host: host(approved()) });
    await userEvent.click(button("Revoke call permissions"));
    expect(
      screen.getByText(/does not delete sources or uninstall the connection/),
    ).toBeTruthy();
    expect(api.revoke).not.toHaveBeenCalled();
    await userEvent.click(
      check("I confirm revoking the call authorization for this install"),
    );
    await userEvent.click(button("Confirm revocation"));
    expect(api.revoke).toHaveBeenCalledExactlyOnceWith("workbuddy", {
      installationId: "isolated-installation-a",
      expectedRevision: 2,
      userConfirmed: true,
    });
    expect(api.set).not.toHaveBeenCalled();
    expect(api.install).not.toHaveBeenCalled();
    expect(
      screen.getByText(
        "The call authorization for this install has been revoked.",
      ),
    ).toBeTruthy();
  });

  it("retains failed revoke inline and does not repeat it without refreshing and confirming again", async () => {
    const api = setup({ host: host(approved()) });
    api.revoke.mockRejectedValueOnce(new Error("unavailable"));
    await userEvent.click(button("Revoke call permissions"));
    await userEvent.click(
      check("I confirm revoking the call authorization for this install"),
    );
    await userEvent.click(button("Confirm revocation"));
    expect(screen.getByRole("alert").textContent).toContain(
      "The revocation result is not confirmed yet",
    );
    expect(button("Confirm revocation").disabled).toBe(true);
    await userEvent.click(button("Refresh and check"));
    expect(api.revoke).toHaveBeenCalledTimes(1);
    await userEvent.click(
      check("I confirm revoking the call authorization for this install"),
    );
    await userEvent.click(button("Confirm revocation"));
    expect(api.revoke).toHaveBeenCalledTimes(2);
  });

  it.each(["none", "get", "set", "revoke"])(
    "is honest and disabled with missing %s bridge methods",
    async (missing) => {
      const api = setup();
      if (missing === "none") delete window.oneMemory;
      if (missing === "get") delete window.oneMemory?.getRecallPermission;
      if (missing === "set") delete window.oneMemory?.setRecallPermission;
      if (missing === "revoke") delete window.oneMemory?.revokeRecallPermission;
      api.update({});
      expect(
        screen.getByText(/does not support full permission management/),
      ).toBeTruthy();
      expect(button("Allow memory use").disabled).toBe(true);
      fireEvent.click(button("Allow memory use"));
      expect(screen.queryByRole("dialog")).toBeNull();
      expect(api.set).not.toHaveBeenCalled();
    },
  );

  it.each([
    permission({ receipt: null }),
    permission({ reason: "installation_incomplete" }),
    permission({ state: "relink-required", reason: "legacy_installation" }),
    permission({ state: "not-installed" }),
    permission({ state: "unavailable" }),
    permission({ state: "unsupported" }),
    permission({ receipt: { ...baseReceipt(), revision: 0 } }),
    permission({ receipt: { ...baseReceipt(), hostId: "cursor" } }),
  ])("blocks incomplete or unusable permission metadata: %j", (value) => {
    const api = setup({ host: host(value) });
    expect(button("Allow memory use").disabled).toBe(true);
    expect(api.get).not.toHaveBeenCalled();
  });

  it("blocks an old shell's missing permission field, and refreshes only on request", async () => {
    const old = host();
    delete old.permission;
    const api = setup({ host: old });
    expect(button("Allow memory use").disabled).toBe(true);
    expect(api.get).not.toHaveBeenCalled();
    await userEvent.click(button("Refresh call permissions"));
    expect(api.get).toHaveBeenCalledTimes(1);
    expect(button("Allow memory use").disabled).toBe(false);
  });

  it("rechecks disabled during async preflight and never commits after being disabled", async () => {
    const api = setup();
    const read = deferred<RecallPermissionStatus>();
    api.get.mockReturnValueOnce(read.promise);
    await open();
    await confirm();
    api.update({ disabled: true });
    await act(async () => read.resolve(permission()));
    expect(api.set).not.toHaveBeenCalled();
    expect(button("Allow and save").disabled).toBe(true);
  });

  it("never reapproves a terminal revoked installation", () => {
    const api = setup({
      host: host(
        permission({
          state: "revoked",
          receipt: { ...baseReceipt(), revision: 3, status: "revoked" },
        }),
      ),
    });
    expect(button("Allow memory use").disabled).toBe(true);
    expect(screen.getByText(/reconnect and grant it again/)).toBeTruthy();
    expect(api.get).not.toHaveBeenCalled();
  });

  it.each(["allowedTools", "allowedScopes"])(
    "does not normalize missing %s into a complete receipt",
    (field) => {
      const partial = { ...baseReceipt() } as unknown as Record<
        string,
        unknown
      >;
      delete partial[field];
      const api = setup({
        host: host(
          permission({
            receipt: partial as unknown as RecallPermissionReceipt,
          }),
        ),
      });
      expect(button("Allow memory use").disabled).toBe(true);
      expect(api.get).not.toHaveBeenCalled();
    },
  );

  it("invalidates the frozen confirmation when the host installation disappears", async () => {
    const api = setup();
    await open();
    api.update({ host: { ...host(), installed: false } });
    expect(screen.getByRole("alert").textContent).toContain("Close and reopen");
    expect(button("Allow and save").disabled).toBe(true);
    expect(api.set).not.toHaveBeenCalled();
  });

  it("retains existing matters permission without offering it to a new installation", async () => {
    const api = setup({ host: host(approved(["om_recall", "om_matters"])) });
    await userEvent.click(button("Adjust call permissions"));
    expect(check("Matters (already authorized)").checked).toBe(true);
    await confirm();
    expect(api.set.mock.calls[0][1].allowedTools).toEqual([
      "om_recall",
      "om_matters",
    ]);
    await userEvent.click(button("Done"));
    api.unmount();
    setup();
    await open();
    expect(screen.queryByRole("checkbox", { name: /Matters/ })).toBeNull();
  });

  it("does not offer an unverified or replaced source identity", async () => {
    setup({
      sources: [
        { ...source(), grant_revision: undefined },
        { ...source("cursor"), candidate_root_id: "new-root" },
      ],
    });
    await open();
    expect(
      screen.queryByRole("checkbox", { name: "WorkBuddy raw conversations" }),
    ).toBeNull();
    expect(
      screen.queryByRole("checkbox", { name: "Cursor raw conversations" }),
    ).toBeNull();
  });

  it("does not override a newer installation prop with a late manual status read", async () => {
    const api = setup();
    const late = deferred<RecallPermissionStatus>();
    api.get.mockReturnValueOnce(late.promise);
    await userEvent.click(button("Refresh call permissions"));
    const next = permission({
      receipt: { ...baseReceipt(), installationId: "isolated-installation-b" },
    });
    api.update({ host: host(next) });
    await act(async () => late.resolve(permission()));
    await open();
    api.get.mockResolvedValue(next);
    await confirm();
    expect(api.set.mock.calls[0][1].installationId).toBe(
      "isolated-installation-b",
    );
  });

  it("owns a body portal, contains focus, restores prior inert state and trigger focus", async () => {
    const background = document.createElement("button");
    background.inert = false;
    const existing = document.createElement("div");
    existing.inert = true;
    document.body.append(background, existing);
    try {
      const api = setup();
      const trigger = button("Adjust scope");
      await open();
      const dialog = screen.getByRole("dialog");
      expect(api.container.contains(dialog)).toBe(false);
      expect(dialog.parentElement?.parentElement).toBe(document.body);
      expect(background.inert).toBe(true);
      expect(api.container.inert).toBe(true);
      expect(document.activeElement).toBe(
        within(dialog).getByRole("button", { name: "Close" }),
      );
      // Tab wraps inside the sheet: back from the first control to the last,
      // forward from the last back to the first.
      await userEvent.tab({ shift: true });
      expect(document.activeElement).toBe(button("Allow and save"));
      await userEvent.tab();
      expect(document.activeElement).toBe(button("Close"));
      background.focus();
      expect(dialog.contains(document.activeElement)).toBe(true);
      await userEvent.keyboard("{Escape}");
      expect(background.inert).toBe(false);
      expect(existing.inert).toBe(true);
      expect(document.activeElement).toBe(trigger);
    } finally {
      background.remove();
      existing.remove();
    }
  });

  it("retains a saved old-installation receipt but requires reopening if identity changes during commit", async () => {
    const api = setup();
    const write = deferred<RecallPermissionStatus>();
    api.set.mockReturnValueOnce(write.promise);
    await open();
    await confirm();
    api.update({
      host: host(
        permission({
          receipt: {
            ...baseReceipt(),
            installationId: "isolated-installation-b",
          },
        }),
      ),
    });
    await act(async () => write.resolve(approvedAll()));
    expect(screen.getByText("The call authorization was saved.")).toBeTruthy();
    expect(screen.getByRole("alert").textContent).toContain("Close and reopen");
    expect(screen.getByText("isolated-installation-a")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Allow and save" })).toBeNull();
    expect(api.set).toHaveBeenCalledTimes(1);
  });

  it("does not treat its own pending committed receipt arriving via parent props as a different identity", async () => {
    const api = setup();
    const write = deferred<RecallPermissionStatus>();
    api.set.mockReturnValueOnce(write.promise);
    await open();
    await confirm();
    api.update({ host: host(approvedAll()) });
    await act(async () => write.resolve(approvedAll()));
    expect(screen.getByText("The call authorization was saved.")).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(api.set).toHaveBeenCalledTimes(1);
  });

  it("rejects ambiguous duplicate source IDs instead of displaying two choices for one authority", async () => {
    setup({ sources: [source(), { ...source(), root_id: "different-root" }] });
    await open();
    expect(
      screen.queryByRole("checkbox", { name: "WorkBuddy raw conversations" }),
    ).toBeNull();
  });

  it("restores the actual trigger even when focus was lost before the dialog effect", async () => {
    setup();
    const trigger = button("Adjust scope");
    expect(document.activeElement).not.toBe(trigger);
    fireEvent.click(trigger);
    await userEvent.click(button("Cancel"));
    expect(document.activeElement).toBe(trigger);
  });

  it("uses a focusable control fallback after revocation removes the initiating button", async () => {
    setup({ host: host(approved()) });
    await userEvent.click(button("Revoke call permissions"));
    await userEvent.click(
      check("I confirm revoking the call authorization for this install"),
    );
    await userEvent.click(button("Confirm revocation"));
    await userEvent.click(button("Done"));
    expect(document.activeElement).toBe(
      screen.getByRole("region", { name: "WorkBuddy call permissions" }),
    );
  });

  it("blocks a selected source ID that becomes ambiguous while the sheet is open", async () => {
    const api = setup();
    await open();
    api.update({
      sources: [source(), { ...source(), root_id: "different-root" }],
    });
    expect(button("Allow and save").disabled).toBe(true);
    expect(screen.getByRole("alert").textContent).toContain("Close and reopen");
    expect(api.set).not.toHaveBeenCalled();
  });

  it.each([
    "host_grant_revision_conflict",
    "host_grant_revoked",
    "host_permission_installation_mismatch",
    "host_permission_relink_required",
  ])("requires reopening after bounded IPC error %s", async (code) => {
    const api = setup();
    api.set.mockRejectedValueOnce(
      new Error(`Error invoking remote method: ${code}`),
    );
    await open();
    await confirm();
    expect(screen.getByRole("alert").textContent).toContain("Close and reopen");
    expect(button("Allow and save").disabled).toBe(true);
    expect(
      screen.queryByRole("button", { name: "Refresh and check" }),
    ).toBeNull();
  });
});
