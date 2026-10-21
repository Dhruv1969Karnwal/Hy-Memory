/** Explicit, separately-confirmed deletion of the One Memory Home.
 *
 * The contract is strict and deliberately awkward, because this is the one
 * action that destroys user memory:
 *   - It is a separate command (`memhub delete-data`), never folded into
 *     `uninstall`. Uninstall keeps the data; this is the only thing that
 *     removes it.
 *   - Preview first, then two independent confirmations: `--yes` alone is not
 *     enough, the caller must also repeat the exact data directory with
 *     `--confirm <path>`.
 *   - Refuse to act unless the target is provably a One Memory Home: it must be
 *     an absolute, non-root, non-home directory that contains the runtime's own
 *     artifacts. An unknown or empty directory is never deleted.
 *   - A live Runtime must be stopped first; deletion while Core holds the
 *     SQLite file would corrupt or silently skip data.
 */

import { lstat, readdir, rm, rmdir } from "node:fs/promises";
import path from "node:path";

export interface DataDeletionPlan {
  schema_version: 1;
  product: string;
  command: "delete-data";
  state: "preview" | "blocked";
  data_dir: string;
  /** Files/dirs that would be removed, for a legible preview. */
  entries: string[];
  bytes: number;
  code?: string;
  reason?: string;
  next_step?: string;
}

export interface DataDeletionReceipt {
  schema_version: 1;
  product: string;
  command: "delete-data";
  state: "deleted";
  data_dir: string;
  removed_entries: string[];
}

/** Markers that prove a directory is a One Memory Home, not an arbitrary path.
 * At least one must exist; a lone directory name is not evidence. */
const HOME_MARKERS = [
  "one-memory.sqlite3",
  "runtime-service.json",
  "host-permissions",
  "hy-memory",
  "integrations",
  "logs",
] as const;

async function directoryInfo(target: string) {
  const info = await lstat(target).catch(() => null);
  return info;
}

export async function planDataDeletion(options: {
  dataDir: string;
  home: string;
  runtimeRunning: boolean;
}): Promise<DataDeletionPlan> {
  const base = (): Pick<
    DataDeletionPlan,
    "schema_version" | "product" | "command" | "entries" | "bytes"
  > => ({
    schema_version: 1,
    product: "memhub",
    command: "delete-data",
    entries: [],
    bytes: 0,
  });
  const blocked = (
    code: string,
    reason: string,
    nextStep: string,
  ): DataDeletionPlan => ({
    ...base(),
    state: "blocked",
    data_dir: options.dataDir,
    code,
    reason,
    next_step: nextStep,
  });
  const target = options.dataDir;
  if (!path.isAbsolute(target) || /[\0\r\n]/.test(target)) {
    return blocked(
      "data_dir_unusable",
      "The data directory must be an absolute path without newlines.",
      "Retry with an absolute path passed to --data-dir.",
    );
  }
  const resolved = path.resolve(target);
  if (resolved === path.parse(resolved).root) {
    return blocked(
      "data_dir_refused",
      "Refusing to delete the filesystem root.",
      "Retry pointing at a specific One Memory Home.",
    );
  }
  if (resolved === path.resolve(options.home)) {
    return blocked(
      "data_dir_refused",
      "Refusing to delete the user's home directory itself.",
      "Retry pointing at the One Memory Home (default ~/.one-memory).",
    );
  }
  if (options.runtimeRunning) {
    return blocked(
      "runtime_still_running",
      "The Runtime is still running; Core holds the database files.",
      "Run memhub stop to stop the Runtime first, then run delete-data.",
    );
  }
  const info = await directoryInfo(resolved);
  if (info === null) {
    return blocked(
      "data_dir_missing",
      "The data directory does not exist.",
      "Confirm --data-dir and retry; nothing to delete.",
    );
  }
  if (!info.isDirectory() || info.isSymbolicLink()) {
    return blocked(
      "data_dir_refused",
      "The data directory is not a plain directory (or is a symlink).",
      "Retry pointing at a real One Memory Home directory.",
    );
  }
  const names = await readdir(resolved).catch(() => [] as string[]);
  const markers = names.filter((name) =>
    (HOME_MARKERS as readonly string[]).includes(name),
  );
  if (markers.length === 0) {
    return blocked(
      "not_a_one_memory_home",
      "The directory contains no One Memory runtime artifacts; refusing to delete an unknown directory.",
      "Confirm --data-dir points at the real One Memory Home.",
    );
  }
  const entries: string[] = [];
  let bytes = 0;
  for (const name of names) {
    const entry = path.join(resolved, name);
    const entryInfo = await lstat(entry).catch(() => null);
    if (entryInfo === null) continue;
    entries.push(entry);
    bytes += await approxBytes(entry);
  }
  entries.sort();
  return { ...base(), state: "preview", data_dir: resolved, entries, bytes };
}

export async function applyDataDeletion(
  plan: DataDeletionPlan,
  confirmation: string,
): Promise<DataDeletionReceipt> {
  if (plan.state !== "preview") {
    throw new Error(plan.code ?? "data_deletion_blocked");
  }
  // The second confirmation must name the exact directory from the preview, so
  // a stale or copy-pasted `--yes` cannot trigger deletion of a different path.
  if (path.resolve(confirmation) !== path.resolve(plan.data_dir)) {
    throw new Error("data_deletion_confirmation_mismatch");
  }
  const removed: string[] = [];
  for (const entry of plan.entries) {
    // Re-check containment at apply time: a plan is data and can be stale.
    if (path.dirname(path.resolve(entry)) !== path.resolve(plan.data_dir)) {
      throw new Error("data_deletion_path_unsafe");
    }
    await rm(entry, { recursive: true, force: true });
    removed.push(entry);
  }
  // Remove the now-empty home itself, but only if it is genuinely empty.
  const remaining = await readdir(plan.data_dir).catch(() => null);
  if (remaining !== null && remaining.length === 0) {
    // `rmdir` on an empty directory; never a recursive delete here, so a race
    // that refilled the directory cannot be silently destroyed.
    await rmdir(plan.data_dir).catch(() => undefined);
  }
  return {
    schema_version: 1,
    product: "memhub",
    command: "delete-data",
    state: "deleted",
    data_dir: plan.data_dir,
    removed_entries: removed,
  };
}

async function approxBytes(target: string): Promise<number> {
  const info = await lstat(target).catch(() => null);
  if (info === null) return 0;
  if (info.isFile()) return info.size;
  if (!info.isDirectory()) return 0;
  let total = 0;
  const names = await readdir(target).catch(() => [] as string[]);
  for (const name of names) {
    total += await approxBytes(path.join(target, name));
  }
  return total;
}
