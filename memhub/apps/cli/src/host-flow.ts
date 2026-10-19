import {
  createHostPermissionClient,
  type HostPermissionClient,
  isHostPresent,
  recallHosts,
} from "@one-memory/runtime-control";
import { countLabel } from "./count-label.js";
import { PRODUCT } from "./version.js";

/** Operations the Runtime exposes for one host, by name.
 *
 * Kept as strings because the connection client passes them through to the
 * Runtime rather than calling them locally: the layouts, the writes to host
 * config files and the permission ledger all stay in Runtime, which is what
 * lets the CLI stay a caller with no window.
 */
export type HostOperation =
  | "getPermission"
  | "previewInstall"
  | "install"
  | "previewUninstall"
  | "uninstall"
  | "previewSkillSync"
  | "applySkillSync"
  | "getSyncHookStatus"
  | "repairInstalledCommand";

/** Operations that change a host's files or its permission ledger. */
const WRITE_OPERATIONS = new Set<HostOperation>([
  "install",
  "uninstall",
  "applySkillSync",
  "repairInstalledCommand",
]);

export function isWriteOperation(operation: HostOperation): boolean {
  return WRITE_OPERATIONS.has(operation);
}

export interface HostDescriptor {
  host_id: string;
  label: string;
  present: boolean;
}

export type HostFlowDocument =
  | {
      schema_version: 1;
      product: string;
      command: "list";
      hosts: HostDescriptor[];
    }
  | {
      schema_version: 1;
      product: string;
      command: "status" | "preview" | "apply";
      host_id: string;
      operation: HostOperation;
      result: unknown;
    }
  | {
      schema_version: 1;
      product: string;
      command: "list" | "status" | "preview" | "apply";
      host_id?: string;
      state: "invalid" | "unavailable";
      code: string;
    };

function failure(
  command: "list" | "status" | "preview" | "apply",
  state: "invalid" | "unavailable",
  code: string,
  hostId?: string,
): HostFlowDocument {
  return {
    schema_version: 1,
    product: PRODUCT,
    command,
    state,
    code,
    ...(hostId ? { host_id: hostId } : {}),
  };
}

/** The hosts on this machine, in the order the app lists them.
 *
 * Only hosts whose own directory exists are returned, so the list describes
 * the machine rather than a roadmap — and `present` is reported rather than
 * assumed, because a host can be configured without being installed.
 */
export async function listHosts(
  hostHome: string,
  environment: NodeJS.ProcessEnv = process.env,
): Promise<HostFlowDocument> {
  const descriptors = await recallHosts(hostHome, {
    ...environment,
    ONE_MEMORY_HOST_HOME: hostHome,
  });
  const present = await Promise.all(
    descriptors.map(async (host) => ({
      host_id: host.hostId,
      label: host.label,
      present: await isHostPresent(host),
    })),
  );
  return {
    schema_version: 1,
    product: PRODUCT,
    command: "list",
    hosts: present,
  };
}

/** One host operation, dispatched by name onto the typed client.
 *
 * The Runtime owns the layouts and every write to a host's config, so this is
 * a passthrough: the CLI never reads or edits host files itself. What it does
 * own is the two-phase shape — see `approvalFingerprint`.
 */
async function hostCall(
  client: HostPermissionClient,
  hostId: string,
  operation: HostOperation,
  args: unknown[],
): Promise<unknown> {
  // The layout is rebuilt inside Runtime from the host id; only the id crosses
  // this boundary, which is why a bare object is enough here.
  const layout = { hostId } as Parameters<
    HostPermissionClient["getPermission"]
  >[0];
  const [approved, overwriteModified] = args;
  switch (operation) {
    case "getPermission":
      return client.getPermission(layout);
    case "previewInstall":
      return client.previewInstall(layout);
    case "install":
      return client.install(layout, approved);
    case "previewUninstall":
      return client.previewUninstall(layout);
    case "uninstall":
      return client.uninstall(layout, approved);
    case "previewSkillSync":
      return client.previewSkillSync(layout);
    case "applySkillSync":
      // `overwriteModified` replaces a skill the user has edited locally. It
      // has to be passed explicitly rather than defaulted: quietly clobbering
      // someone's edited skill is how a working integration disappears.
      return client.applySkillSync(
        layout,
        approved,
        overwriteModified === true,
      );
    case "getSyncHookStatus":
      return client.getSyncHookStatus(layout);
    case "repairInstalledCommand":
      return client.repairInstalledCommand(layout);
  }
}

export async function readHost(
  dataDir: string,
  hostId: string,
  operation: HostOperation,
  args: unknown[] = [],
): Promise<HostFlowDocument> {
  // Matches the CLI's parse: a profile suffix (`:web`) is part of a real host
  // id, so it has to be accepted here too.
  if (!/^[a-z0-9][a-z0-9-]{0,31}(:[a-z0-9][a-z0-9-]{0,31})?$/.test(hostId)) {
    return failure("status", "invalid", "invalid_host_id", hostId);
  }
  const client = createHostPermissionClient({ dataDir });
  const result = await hostCall(client, hostId, operation, args);
  if (result === undefined) {
    return failure(
      "status",
      "unavailable",
      "host_operation_unavailable",
      hostId,
    );
  }
  return {
    schema_version: 1,
    product: PRODUCT,
    command: isWriteOperation(operation) ? "apply" : "status",
    host_id: hostId,
    operation,
    result,
  };
}

/** The preview fingerprint that an install or uninstall must present.
 *
 * `install` and `uninstall` refuse without it (`host_install_confirmation_required`),
 * and they compare it against a freshly computed preview rather than trusting
 * the caller — so the thing being approved is what the host looks like now,
 * not what it looked like when the preview was printed.
 */
export function approvalFingerprint(result: unknown): string | null {
  if (typeof result !== "object" || result === null) return null;
  const fingerprint = (result as Record<string, unknown>).fingerprint;
  return typeof fingerprint === "string" && /^[0-9a-f]{64}$/.test(fingerprint)
    ? fingerprint
    : null;
}

export function hostExitCode(document: HostFlowDocument): number {
  if ("state" in document) {
    return document.state === "invalid" ? 2 : 1;
  }
  return 0;
}

export function renderHostText(document: HostFlowDocument): string {
  if ("state" in document) {
    const lines = [
      document.state === "invalid"
        ? "Invalid arguments."
        : `Unavailable: ${document.code}`,
    ];
    if (document.host_id) lines.push(`  host  ${document.host_id}`);
    return `${lines.join("\n")}\n`;
  }
  if (document.command === "list") {
    if (document.hosts.length === 0) {
      return "No connectable hosts found on this machine.\n";
    }
    const lines = document.hosts.map(
      (host) =>
        `- ${host.label}: ${host.present ? "connected" : "not connected"}`,
    );
    return `Connectable apps:\n${lines.join("\n")}\n`;
  }
  const lines = [`host ${document.host_id} · ${document.operation}`];
  const result = document.result as Record<string, unknown> | null;
  if (result && typeof result === "object") {
    if (typeof result.state === "string")
      lines.push(`  state  ${result.state}`);
    if (result.reason !== null && result.reason !== undefined) {
      lines.push(`  reason ${String(result.reason)}`);
    }
    if (Array.isArray(result.files)) {
      lines.push(`  files  ${result.files.length}`);
      for (const file of result.files.slice(0, 8)) {
        const entry = file as Record<string, unknown>;
        lines.push(`    - ${String(entry.path)}`);
      }
      if (result.files.length > 8) {
        lines.push(`    … and ${result.files.length - 8} more`);
      }
    }
    if (typeof result.fingerprint === "string") {
      lines.push(`  fingerprint  ${result.fingerprint}`);
      lines.push("");
      lines.push(
        "This is a preview; no files were changed. To confirm, run: host install --host <id> --approve <fingerprint>",
      );
    }
    // Shown because the Runtime refuses to uninstall while anything still
    // points at us from a file it does not own: those are other people's
    // hooks, and removing them is the user's decision, not ours. Without this
    // the refusal looks like a bug rather than a request to look.
    const review = result.review;
    if (Array.isArray(review) && review.length > 0) {
      lines.push(
        `  review  ${countLabel(review.length, "item")} still need manual review:`,
      );
      for (const item of review.slice(0, 12)) {
        const entry = item as Record<string, unknown>;
        lines.push(`    - ${String(entry.path)} :: ${String(entry.where)}`);
      }
      if (review.length > 12)
        lines.push(`    … and ${review.length - 12} more`);
      lines.push("");
      lines.push(
        "These were not installed by us; uninstalling will not delete them for you. Remove them yourself before uninstalling.",
      );
    }
  }
  return `${lines.join("\n")}\n`;
}
