import { MEMORY_STATUSES } from "./core-responses.js";
import {
  EXIT_FAILED,
  EXIT_OK,
  EXIT_PLAN_BLOCKED,
  EXIT_RUNTIME_STOPPED,
  EXIT_USAGE,
} from "./exit-codes.js";

export type CommandOption = {
  readonly flag: string;
  readonly summary: string;
};

export type CommandDescriptor = {
  readonly name: string;
  readonly summary: string;
  readonly usage: string;
  readonly alternative_usage?: string;
  readonly capability_aliases?: readonly string[];
  readonly options: readonly CommandOption[];
  readonly exit_codes: readonly number[];
};

/**
 * Product-level names that are intentionally represented by an existing
 * command rather than by a second command with duplicate Runtime semantics.
 * Keep this machine-readable so package acceptance can compare the declared
 * CLI surface to the product lifecycle requirements.
 */
export const CAPABILITY_ALIASES: Readonly<Record<string, readonly string[]>> = {
  daemon: ["start", "stop", "restart", "status"],
  "startup-service": ["service-plan", "service-install", "service-uninstall"],
};

const PLATFORM_OPTION: CommandOption = {
  flag: "--platform <darwin|linux|win32>",
  summary: "Target platform; defaults to the current platform.",
};
const HOME_OPTION: CommandOption = {
  flag: "--home <path>",
  summary: "User home directory; defaults to the current user's home.",
};
const DATA_DIR_OPTION: CommandOption = {
  flag: "--data-dir <path>",
  summary:
    "Absolute path of the MemHub data directory; reuses ONE_MEMORY_DATA_DIR by default and never rewrites existing data.",
};
const UID_OPTION: CommandOption = {
  flag: "--uid <id>",
  summary:
    "User ID that owns the macOS LaunchAgent; defaults to the current user.",
};
const XDG_OPTION: CommandOption = {
  flag: "--xdg-config-home <path>",
  summary: "Root directory of the Linux systemd user configuration.",
};
const CONFIRM_OPTION: CommandOption = {
  flag: "--yes",
  summary:
    "Confirm execution; without it only a plan is printed and nothing changes.",
};
const JSON_OPTION: CommandOption = {
  flag: "--json",
  summary: "Print JSON to stdout for scripts and Agents to consume.",
};
const WATCH_OPTION: CommandOption = {
  flag: "--watch",
  summary:
    "Serial continuous refresh; Ctrl-C only stops watching, not the Runtime.",
};
const INTERVAL_OPTION: CommandOption = {
  flag: "--interval <seconds>",
  summary:
    "Refresh interval, default 5 seconds, range 1..300; must be used with --watch.",
};
const PLAN_FILE_OPTION: CommandOption = {
  flag: "--plan-file <path>",
  summary: "Read a bounded plan file saved by an earlier --json run.",
};
const EXPLICIT_CONFIRM_OPTION: CommandOption = {
  flag: "--confirm",
  summary:
    "Confirm executing exactly what the plan file describes; without it nothing is ever written.",
};
// Same flag, but for commands that name their target directly instead of
// going through a saved plan file; the plan-file wording would be wrong here.
const DIRECT_CONFIRM_OPTION: CommandOption = {
  flag: "--confirm",
  summary:
    "Confirm executing the operation previewed above; without it nothing is ever written.",
};

const SERVICE_OPTIONS: readonly CommandOption[] = [
  PLATFORM_OPTION,
  HOME_OPTION,
  {
    flag: "--install-root <path>",
    summary: "Managed install root (required).",
  },
  DATA_DIR_OPTION,
  UID_OPTION,
  XDG_OPTION,
];

const SERVICE_USAGE =
  "--install-root <path> [--platform <id>] [--home <path>] [--data-dir <path>] [--uid <id>] [--xdg-config-home <path>]";

const INSTALL_OPTIONS: readonly CommandOption[] = [
  {
    flag: "--source <path>",
    summary:
      "Offline release directory; when it is given there is no network access at all. The CLI inside a full Runtime derives the same directory by default; the npm CLI downloads the Release of the same version.",
  },
  {
    flag: "--release-manifest-url <https-url>",
    summary:
      "Remote release-manifest.json; used only when --source is not given. A release build may inject a default template of the same version; a source build assumes no private download source.",
  },
  {
    flag: "--channel <stable|prerelease>",
    summary:
      "Remote release channel; a SemVer prerelease defaults to prerelease and everything else to stable. It never crosses channels silently.",
  },
  {
    flag: "--install-root <path>",
    summary: "Managed install root; derived from the platform by default.",
  },
  HOME_OPTION,
  {
    flag: "--bin-dir <path>",
    summary:
      "Directory holding the managed CLI links; on POSIX it defaults to ~/.local/bin.",
  },
  DATA_DIR_OPTION,
  UID_OPTION,
  XDG_OPTION,
  {
    flag: "--no-shell-path",
    summary:
      "Do not touch the login shell profile (for isolated or scripted installs); short commands need --bin-dir added to PATH by you.",
  },
  {
    flag: "--shell-profile <path>",
    summary:
      "Write the PATH block to this file instead of the default ~/.zprofile / ~/.profile, which makes isolated installs easier.",
  },
  {
    flag: "--preserve-data-dir",
    summary:
      "For uninstall: when --data-dir sits inside <install-root>/versions/, state explicitly that this run does not treat that directory as a managed subdirectory; it only changes what the plan protects and never deletes data.",
  },
  {
    flag: "--allow-downgrade",
    summary:
      "For install/upgrade: explicitly allow installing a release package below the current build_number; refused by default. No env var or configuration turns it on by default.",
  },
  CONFIRM_OPTION,
];

const INSTALL_USAGE =
  "[--source <path> | --release-manifest-url <https-url>] [--channel <stable|prerelease>] [--install-root <path>] [--home <path>] [--bin-dir <path>] [--data-dir <path>] [--uid <id>] [--xdg-config-home <path>] [--no-shell-path] [--shell-profile <path>] [--preserve-data-dir] [--allow-downgrade] [--yes]";

export const COMMANDS: readonly CommandDescriptor[] = [
  {
    name: "serve",
    summary:
      "Run the full Runtime (Core and Host Gateway) in the foreground without registering a system service.",
    usage: "memhub serve",
    options: [],
    exit_codes: [EXIT_OK, EXIT_FAILED],
  },
  {
    name: "start",
    summary:
      "Start the local Runtime; when an instance that passes identity verification already exists it only connects instead of starting another one.",
    usage: "memhub start [--json]",
    options: [JSON_OPTION],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_RUNTIME_STOPPED],
  },
  {
    name: "stop",
    summary:
      "Stop the local Runtime that passed identity verification; no signal is sent when the identity is unclear.",
    usage: "memhub stop [--json]",
    options: [JSON_OPTION],
    exit_codes: [EXIT_OK, EXIT_FAILED],
  },
  {
    name: "restart",
    summary: "Stop and then start the local Runtime, as two sequential steps.",
    usage: "memhub restart [--json]",
    options: [JSON_OPTION],
    exit_codes: [EXIT_OK, EXIT_FAILED],
  },
  {
    name: "status",
    summary:
      "Read-only Runtime status query; it starts no service, migrates no data and changes no configuration.",
    usage: "memhub status [--watch] [--interval <seconds>] [--json]",
    options: [WATCH_OPTION, INTERVAL_OPTION, JSON_OPTION],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_RUNTIME_STOPPED],
  },
  {
    name: "doctor",
    summary:
      "The same read-only checks under the diagnostic rubric; nothing is repaired by default.",
    usage: "memhub doctor [--json]",
    options: [JSON_OPTION],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_RUNTIME_STOPPED],
  },
  {
    name: "logs",
    summary:
      "Read the local redacted diagnostic log; logs export saves the troubleshooting bundle straight into the downloads directory. Shows the last 200 events of the most recent session by default.",
    usage:
      "memhub logs export [--json]\nmemhub logs [--since <15m|2h|7d|RFC3339>] [--session <id>] [--trace <id>] [--incident <id>] [--operation <id>] [--request <id>] [--job <id>] [--task <id>] [--component <name>] [--event <name>] [--errors-only] [--tail <1..5000>] [--follow] [--json]",
    options: [
      {
        flag: "--since <time>",
        summary: "Read from a relative duration or an RFC3339 timestamp.",
      },
      { flag: "--session <id>", summary: "Show only one runtime session." },
      {
        flag: "--trace <id>",
        summary: "Show only one cross-component operation.",
      },
      {
        flag: "--incident <id>",
        summary: "Show events from the same instance around the incident time.",
      },
      {
        flag: "--operation <id>",
        summary: "Show only one persisted operation.",
      },
      {
        flag: "--request <id>",
        summary: "Show only one model or write request.",
      },
      { flag: "--job <id>", summary: "Show only one import job." },
      { flag: "--task <id>", summary: "Show only one memory processing task." },
      { flag: "--component <name>", summary: "Filter by component." },
      { flag: "--event <name>", summary: "Filter by event name." },
      { flag: "--errors-only", summary: "Show error events only." },
      {
        flag: "--tail <n>",
        summary:
          "Maximum number of entries to return, default 200, capped at 5000.",
      },
      {
        flag: "--follow",
        summary: "Keep reading new events; Ctrl-C only stops watching.",
      },
      {
        flag: "--json",
        summary:
          "Print NDJSON to stdout (one redacted event per line; not a single JSON document) for scripts and Agents to consume.",
      },
    ],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_USAGE],
  },
  {
    name: "init",
    summary:
      "First connection: pick sources, then decide whether to import history. Connecting turns on processing for that Harness; use sync pause/resume afterwards to freeze or resume it. In a terminal without --source it enters the interactive wizard (everything selected by default); without a terminal or with --json it reads no stdin and only lists the available sources and the next step. --all-detected connects every detected source at once.",
    usage:
      "memhub init [--json]\n            memhub init --source <id> [--source <id> ...] [--import] [--dry-run] [--full-rescan] [--confirm-overrides] [--json]\n            memhub init --all-detected [--import] [--dry-run] [--json]",
    options: [
      {
        flag: "--source <id>",
        summary:
          "Source to connect; repeatable. In a terminal, leaving it out enters the interactive wizard.",
      },
      {
        flag: "--all-detected",
        summary:
          "Connect, and with --import also import, every source with location_state=found.",
      },
      {
        flag: "--import",
        summary:
          "Import the source's history conversations right after connecting (connecting alone does not import).",
      },
      {
        flag: "--dry-run",
        summary:
          "Plan only: no connecting and no importing, so you can see what it would do.",
      },
      {
        flag: "--full-rescan",
        summary: "Ignore existing cursors and rescan everything during import.",
      },
      {
        flag: "--confirm-overrides",
        summary:
          "Confirm that environment override sources, such as the directory DSH_HOME points at, are accepted.",
      },
      JSON_OPTION,
    ],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_RUNTIME_STOPPED],
  },
  {
    name: "setup",
    summary:
      "Build a headless first-import plan; it reads only existing sources, probes no directory and writes no authorization.",
    usage: "memhub setup plan --source <id> [--source <id> ...] [--json]",
    options: [
      {
        flag: "--source <id>",
        summary:
          "Source to include in the import plan; repeatable 1..32 times.",
      },
      JSON_OPTION,
    ],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_RUNTIME_STOPPED],
  },
  {
    name: "sources",
    summary: "List sources and run confirmed import plans.",
    usage:
      "memhub sources list [--json]\n            memhub sources import --plan-file <path> --confirm [--json]\n            memhub sources receipt <request-id> [--json]",
    options: [PLAN_FILE_OPTION, EXPLICIT_CONFIRM_OPTION, JSON_OPTION],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_RUNTIME_STOPPED],
  },
  {
    name: "sync",
    summary:
      "Show Harness processing state; freeze or thaw a single source directly.",
    usage:
      "memhub sync status [--json]\n            memhub sync <pause|resume> --source <id> [--json]",
    options: [
      {
        flag: "--source <id>",
        summary: "Source to pause or resume.",
      },
      JSON_OPTION,
    ],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_RUNTIME_STOPPED],
  },
  {
    name: "progress",
    summary:
      "Read-only view of import and generation progress; state and counts are reported separately and no completion percentage is invented.",
    usage:
      "memhub progress [--job <job-id>] [--issues] [--limit <n>] [--watch] [--interval <seconds>] [--json]",
    options: [
      { flag: "--job <job-id>", summary: "Show only one import job." },
      {
        flag: "--issues",
        summary: "Also read this import job's issues; must be used with --job.",
      },
      {
        flag: "--limit <n>",
        summary:
          "Maximum entries per job or issue list, default 50, capped at 100.",
      },
      WATCH_OPTION,
      INTERVAL_OPTION,
      JSON_OPTION,
    ],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_RUNTIME_STOPPED],
  },
  {
    name: "jobs",
    summary:
      "Start or control a single import job: start creates a history/sync job; pause/resume/cancel control an existing one. Every write prints a plan first and needs an explicit --confirm; list/show are read-only.",
    usage:
      "memhub jobs list [--json]\n" +
      "            memhub jobs show <job-id> [--json]\n" +
      "            memhub jobs start --source <id> [--operation <history|sync>] [--confirm] [--json]\n" +
      "            memhub jobs <pause|resume|cancel> --job <id> [--confirm] [--json]",
    options: [
      {
        flag: "--source <id>",
        summary:
          "Source that start should import; it must already be connected.",
      },
      {
        flag: "--operation <history|sync>",
        summary: "Import type for start; defaults to history.",
      },
      {
        flag: "--job <id>",
        summary: "Import job targeted by pause/resume/cancel.",
      },
      DIRECT_CONFIRM_OPTION,
      JSON_OPTION,
    ],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_RUNTIME_STOPPED],
  },
  {
    name: "processing",
    summary:
      "Manually advance pending evidence; it previews by default and executes after confirmation.",
    usage:
      "memhub processing extract [--evidence <id> ...] [--confirm] [--json]",
    options: [
      {
        flag: "--evidence <id>",
        summary:
          "Evidence ID that processing extract should handle ahead of the rest; repeatable. Without it the next batch advances.",
      },
      DIRECT_CONFIRM_OPTION,
      JSON_OPTION,
    ],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_RUNTIME_STOPPED],
  },
  {
    name: "models",
    summary:
      "Configure a model interactively, or build configuration/probe plans, execute confirmed operations and query receipts; in script mode credentials are read only from a controlled file.",
    usage:
      "memhub models list [--json]\n            memhub models show <profile-id> [--json]\n            memhub models plan --config-file <path> [--json]\n            memhub models configure --plan-file <path> --confirm [--api-key-file <path>] [--json]\n            memhub models receipt <operation-id> [--json]\n            memhub models edit <profile-id> --config-file <path> [--confirm] [--api-key-file <path>] [--json]\n            memhub models activate <profile-id> [--confirm] [--json]\n            memhub models probe-plan <profile-id> [--json]\n            memhub models probe --plan-file <path> --confirm [--json]\n            memhub models probe-receipt <operation-id> [--json]\n            memhub models remove <profile-id> [--confirm] [--json]",
    alternative_usage: "memhub models setup",
    options: [
      {
        flag: "--config-file <path>",
        summary: "Bounded JSON model configuration without api_key.",
      },
      PLAN_FILE_OPTION,
      EXPLICIT_CONFIRM_OPTION,
      {
        flag: "--api-key-file <path>",
        summary:
          "Read credentials from a regular file; the contents never enter arguments, output or errors.",
      },
      JSON_OPTION,
    ],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_RUNTIME_STOPPED],
  },
  {
    name: "memories",
    summary:
      "List or view memories, or build and explicitly confirm a governance operation plan carrying version and request identity.",
    usage:
      "memhub memories list [--query <text>] [--query-file <path>] [--status <state>] [--source <id>] [--offset <n>] [--limit <n>] [--json]\n            memhub memories show <memory-id> [--json]\n            memhub memories plan add --content-file <path> --scope <scope> --evidence <id> [--kind <kind>] [--json]\n            memhub memories plan <correct|delete|scope> <memory-id> [--content-file <path>] [--scope <scope>] [--json]\n            memhub memories apply --plan-file <path> --confirm [--content-file <path>] [--json]",
    options: [
      {
        flag: "--query <text>",
        summary:
          "Filter by content; it appears in shell history, so use --query-file for sensitive terms.",
      },
      {
        flag: "--query-file <path>",
        summary:
          "Read the filter term from a file so the query never enters command-line history.",
      },
      {
        flag: "--status <state>",
        summary: `Memory status; one of ${MEMORY_STATUSES.join(", ")}.`,
      },
      {
        flag: "--source <id>",
        summary: "Show only memories produced by one source.",
      },
      { flag: "--offset <n>", summary: "Skip the first n entries, default 0." },
      {
        flag: "--limit <n>",
        summary: "Maximum entries per page, default 100, capped at 500.",
      },
      {
        flag: "--content-file <path>",
        summary:
          "Read the add/correct body from a bounded regular file; the plan keeps only a digest and re-validates it at execution.",
      },
      {
        flag: "--kind <chat|working|coding|schema|intention>",
        summary: "Type of the new memory; defaults to chat.",
      },
      {
        flag: "--scope <scope>",
        summary: "Memory use scope; repeatable, and private must be exclusive.",
      },
      {
        flag: "--evidence <id>",
        summary: "Evidence ID referenced by the new memory; repeatable.",
      },
      PLAN_FILE_OPTION,
      EXPLICIT_CONFIRM_OPTION,
      JSON_OPTION,
    ],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_RUNTIME_STOPPED],
  },
  {
    name: "calls",
    summary:
      "Clear local call records; an irreversible delete that plans by default and needs an explicit --confirm to run.",
    usage:
      "memhub calls clear --event <id> [--event <id> ...] [--confirm] [--json]\n            memhub calls clear --all [--confirm] [--json]",
    options: [
      {
        flag: "--event <id>",
        summary: "Call record event ID to clear; repeatable, at most 100.",
      },
      {
        flag: "--all",
        summary: "Clear all call records; mutually exclusive with --event.",
      },
      DIRECT_CONFIRM_OPTION,
      JSON_OPTION,
    ],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_RUNTIME_STOPPED],
  },
  {
    name: "import-file",
    summary:
      "Upload, submit or discard a history import file; submit and discard plan by default and need --confirm.",
    usage:
      "memhub import-file upload --file <path> [--kind raw_conversation|memory_content] [--confirm] [--json]\n            memhub import-file commit <upload-id> [--confirm] [--json]\n            memhub import-file discard <upload-id> [--confirm] [--json]",
    options: [
      {
        flag: "--file <path>",
        summary:
          "Regular file to upload (not a symlink, <=128 MiB, .zip/.json/.md/.txt).",
      },
      {
        flag: "--kind <kind>",
        summary:
          "Input type; defaults to raw_conversation, and memory_content is for memory bodies.",
      },
      DIRECT_CONFIRM_OPTION,
      JSON_OPTION,
    ],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_RUNTIME_STOPPED],
  },
  {
    name: "notes",
    summary:
      "Add memories inside the app (memory-notes): add from a regular file, or process existing notes after an explicit confirmation.",
    usage:
      "memhub notes add --content-file <path> [--json]\n            memhub notes apply --plan-file <path> --content-file <path> --confirm [--json]\n            memhub notes process <note-id> [--confirm] [--json]",
    options: [
      {
        flag: "--content-file <path>",
        summary:
          "Read the new body from a bounded regular file; the plan keeps only a digest and re-validates it at execution.",
      },
      PLAN_FILE_OPTION,
      DIRECT_CONFIRM_OPTION,
      JSON_OPTION,
    ],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_RUNTIME_STOPPED],
  },
  {
    name: "extraction",
    summary:
      "Set the extraction window; it plans by default and writes only after --confirm.",
    usage: "memhub extraction set --window <3m|6m|1y|all> [--confirm] [--json]",
    options: [
      {
        flag: "--window <preset>",
        summary: "Extraction window preset: 3m, 6m, 1y or all.",
      },
      DIRECT_CONFIRM_OPTION,
      JSON_OPTION,
    ],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_RUNTIME_STOPPED],
  },
  {
    name: "graph",
    summary:
      "The user's decision when the graph width does not match; it plans by default and runs only after --confirm.",
    usage:
      "memhub graph width --decision <keep_existing|rebuild_with_new|export_first> [--confirm] [--json]",
    options: [
      {
        flag: "--decision <decision>",
        summary:
          "keep_existing (keep), export_first (export first), rebuild_with_new (export then rebuild).",
      },
      DIRECT_CONFIRM_OPTION,
      JSON_OPTION,
    ],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_RUNTIME_STOPPED],
  },
  {
    name: "host",
    summary:
      "Host integration and being called: list local hosts, view integration state, preview and install or uninstall recall injection. Install and uninstall take two steps: preview to get the fingerprint, then submit it with --approve.",
    usage:
      "memhub host list [--json]\n            memhub host status --host <id> [--json]\n            memhub host install --host <id> [--json]\n            memhub host install --host <id> --approve <fingerprint> [--json]\n            memhub host uninstall --host <id> [--json]\n            memhub host uninstall --host <id> --approve <fingerprint> [--json]\n            memhub host sync skill --host <id> [--approve <fingerprint>] [--overwrite-modified] [--json]",
    alternative_usage:
      "memhub host install <id> [--approve <fingerprint>] [--json]\n            memhub host status <id> [--json]\n            memhub host uninstall <id> [--approve <fingerprint>] [--json]",
    options: [
      {
        flag: "--host <id>",
        summary:
          "Target host, such as cursor, claude-code or codex; it may also follow the verb directly (memhub host install codex). See host list for the complete set.",
      },
      {
        flag: "--approve <fingerprint>",
        summary:
          "Submit the fingerprint the preview returned; without it install/uninstall only preview and change no host file.",
      },
      {
        flag: "--overwrite-modified",
        summary:
          "Allow host sync skill to overwrite a locally modified skill when submitting; it does not overwrite by default.",
      },
      JSON_OPTION,
    ],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_RUNTIME_STOPPED],
  },
  {
    name: "recall",
    summary:
      "Retrieve context from the currently authorized source text and governed memories; local and read-only, calling no model and triggering no index backfill.",
    usage:
      "memhub recall <query> [--kind <raw|memory>] [--source <id>] [--limit <n>] [--character-budget <n>] [--timeout-ms <n>] [--json]\n            memhub recall --query-file <path> [...]\n            memhub recall --stdin [...]",
    options: [
      {
        flag: "--query-file <path>",
        summary:
          "Read the query from a bounded regular file so it never enters shell history.",
      },
      {
        flag: "--stdin",
        summary:
          "Explicitly read one bounded query from a pipe; an interactive TTY is refused and no input is waited on.",
      },
      {
        flag: "--kind <raw|memory>",
        summary:
          "Query path; repeatable, and by default both raw and memory are queried.",
      },
      {
        flag: "--source <id>",
        summary:
          "Narrow the raw source text only; repeatable, and refused when raw was not requested.",
      },
      {
        flag: "--limit <n>",
        summary: "Return at most 1..20 items, default 8.",
      },
      {
        flag: "--character-budget <n>",
        summary: "Total body budget of 200..20000 characters, default 4000.",
      },
      {
        flag: "--timeout-ms <n>",
        summary: "Core query deadline of 50..30000 milliseconds, default 2000.",
      },
      JSON_OPTION,
    ],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_RUNTIME_STOPPED],
  },
  {
    name: "service-plan",
    summary:
      "Build the user-level service plan; it only prints the plan, writes no file and runs no command.",
    usage: `memhub service-plan ${SERVICE_USAGE}`,
    options: SERVICE_OPTIONS,
    exit_codes: [EXIT_OK, EXIT_PLAN_BLOCKED],
  },
  {
    name: "service-install",
    summary:
      "Preview installing the user-level service; only --yes writes the service file and registers it.",
    usage: `memhub service-install ${SERVICE_USAGE} [--yes]`,
    options: [...SERVICE_OPTIONS, CONFIRM_OPTION],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_PLAN_BLOCKED],
  },
  {
    name: "service-uninstall",
    summary:
      "Preview uninstalling the user-level service; only --yes unregisters it and deletes the exactly matching managed service file, and no data is deleted.",
    usage: `memhub service-uninstall ${SERVICE_USAGE} [--yes]`,
    options: [...SERVICE_OPTIONS, CONFIRM_OPTION],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_PLAN_BLOCKED],
  },
  {
    name: "install",
    summary:
      "Preview the release directory install, the memhub/memhub-cli/one-memory shortcuts, the PATH entry and the user service plan; only --yes executes, and it rolls back on failure.",
    usage: `memhub install ${INSTALL_USAGE}`,
    options: INSTALL_OPTIONS,
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_PLAN_BLOCKED],
  },
  {
    name: "uninstall",
    summary:
      "Uninstall the product itself: stop the service and remove the managed CLI links and installed version files; it does **not delete data** (dataDir is kept as is). Only --yes executes; by default it only prints a plan.",
    usage: `memhub uninstall ${INSTALL_USAGE}`,
    options: INSTALL_OPTIONS,
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_PLAN_BLOCKED],
  },
  {
    name: "upgrade",
    summary:
      "Preview the upgrade; only --yes switches the managed version and restarts the service, restores the old version on failure, and deletes no data.",
    usage: `memhub upgrade ${INSTALL_USAGE}`,
    options: INSTALL_OPTIONS,
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_PLAN_BLOCKED],
  },
  {
    name: "delete-data",
    summary:
      "Delete One Memory Home (the data) as a separate action with a second confirmation. It is the only command that deletes data; uninstall does not. It previews by default, requires the Runtime to be stopped, and needs an explicit --confirm <path> restating that directory.",
    usage:
      "memhub delete-data [--data-dir <path>] [--yes --confirm <path>] [--json]",
    options: [
      DATA_DIR_OPTION,
      CONFIRM_OPTION,
      {
        flag: "--confirm <path>",
        summary:
          "Second confirmation: repeat the data directory from the preview verbatim; without it nothing is deleted even with --yes.",
      },
      JSON_OPTION,
    ],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_PLAN_BLOCKED],
  },
  {
    name: "onboarding",
    summary:
      "onboarding reset returns to the first-run setup flow. It clears only the setup completion state: it deletes no memory data and nothing has to be uninstalled and reinstalled.",
    usage: "memhub onboarding reset [--json]",
    options: [JSON_OPTION],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_USAGE],
  },
  {
    name: "help",
    summary: "Show the general help or the usage of one command.",
    usage: "memhub help [command] [--json]",
    options: [JSON_OPTION],
    exit_codes: [EXIT_OK],
  },
  {
    name: "version",
    summary: "Show the CLI version.",
    usage: "memhub version [--json]",
    options: [JSON_OPTION],
    exit_codes: [EXIT_OK],
  },
  {
    name: "path",
    summary:
      "View or register the scanned directories (scan roots). list is read-only; add/remove submit a preview only with --yes; all of them are wired to the Runtime's scan-roots endpoints.",
    usage:
      "memhub path list [--json]\n" +
      "            memhub path add --root <path> [--source <id>] [--no-recursive] [--yes]\n" +
      "            memhub path remove --root <path> [--yes]",
    options: [DATA_DIR_OPTION, JSON_OPTION, CONFIRM_OPTION],
    exit_codes: [EXIT_OK, EXIT_USAGE, EXIT_FAILED, EXIT_PLAN_BLOCKED],
  },
  {
    name: "adapter",
    summary:
      "View a Harness Adapter's readiness and last probe result (list), or probe one source live (probe --source <id>); both are read-only and wired to the Runtime's adapter and probe endpoints.",
    usage:
      "memhub adapter list [--json]\n" +
      "            memhub adapter probe --source <id> [--json]",
    options: [DATA_DIR_OPTION, JSON_OPTION],
    exit_codes: [EXIT_OK, EXIT_USAGE, EXIT_FAILED],
  },
  {
    name: "scan",
    summary:
      "Scan immediately; it prints a plan by default and really runs with --yes. --source scans a single source and --all scans or imports every detected source at once (the Runtime addresses sources and takes no bare path).",
    usage:
      "memhub scan [--source <id>] [--full] [--yes] [--json]\n            memhub scan --all [--yes] [--json]",
    options: [
      {
        flag: "--source <id>",
        summary: "Scan a single source; mutually exclusive with --all.",
      },
      {
        flag: "--all",
        summary:
          "Process every source with location_state=found and report one receipt per source; a partial success returns non-zero.",
      },
      {
        flag: "--full",
        summary:
          "Ignore existing cursors and rescan everything; incremental reads only by default.",
      },
      DATA_DIR_OPTION,
      JSON_OPTION,
      CONFIRM_OPTION,
    ],
    exit_codes: [EXIT_OK, EXIT_USAGE, EXIT_FAILED, EXIT_PLAN_BLOCKED],
  },
  {
    name: "write",
    summary:
      "Submit one memory write and get back a queryable request id; the write completes asynchronously and must not be submitted twice.",
    usage:
      "memhub write --session <id> --from <round> --to <round> --evidence <id> [--evidence <id>...] [--live] [--json]",
    options: [DATA_DIR_OPTION, JSON_OPTION],
    exit_codes: [EXIT_OK, EXIT_USAGE, EXIT_FAILED],
  },
  {
    name: "task",
    summary:
      "Query a write task's status; unknown means the outcome is pending and is no reason to resend.",
    usage: "memhub task --id <request-id> [--json]",
    options: [DATA_DIR_OPTION, JSON_OPTION],
    exit_codes: [EXIT_OK, EXIT_USAGE, EXIT_FAILED],
  },
  {
    name: "backup",
    summary:
      "Back up and restore MemHub data (two phases). create calls the Runtime's backup endpoint; restore overwrites existing data, requires the Runtime to be stopped and `--confirm` to repeat `--source` verbatim, after which a non-Runtime maintenance-mode Core performs the managed verification, the protective backup and the atomic swap.",
    usage:
      "memhub backup create --target <path> [--yes] [--json]\n" +
      "            memhub backup restore --source <path> --yes --confirm <path> [--json]",
    options: [
      DATA_DIR_OPTION,
      JSON_OPTION,
      CONFIRM_OPTION,
      {
        flag: "--confirm <path>",
        summary:
          "restore requires repeating the --source path verbatim; on any mismatch nothing changes.",
      },
    ],
    exit_codes: [EXIT_OK, EXIT_USAGE, EXIT_FAILED, EXIT_PLAN_BLOCKED],
  },
  {
    name: "ui",
    summary:
      "Open the Web Console hosted by the Runtime in the system's default browser: first exchange a one-time loopback login code with Core, then hand it to the browser so Core sets an HttpOnly/SameSite=Strict session; the CLI never exposes a bare Core token to the browser. When Core does not offer that contract it honestly returns ui_session_contract_unavailable (exit 4).",
    usage: "memhub ui [--json]",
    options: [JSON_OPTION],
    exit_codes: [EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_PLAN_BLOCKED],
  },
];

export function findCommand(name: string): CommandDescriptor | undefined {
  return COMMANDS.find((command) => command.name === name);
}
