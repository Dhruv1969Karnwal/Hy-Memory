export const EXIT_OK = 0;
export const EXIT_FAILED = 1;
export const EXIT_USAGE = 2;
export const EXIT_RUNTIME_STOPPED = 3;
export const EXIT_PLAN_BLOCKED = 4;

export type ExitCode = {
  readonly code: number;
  readonly name: string;
  readonly summary: string;
};

export const EXIT_CODES: readonly ExitCode[] = [
  { code: EXIT_OK, name: "ok", summary: "The command completed as requested." },
  {
    code: EXIT_FAILED,
    name: "failed",
    summary:
      "The command failed or its state couldn't be confirmed; don't replay write operations when the result is unknown.",
  },
  {
    code: EXIT_USAGE,
    name: "usage",
    summary: "Invalid arguments or command; nothing was done.",
  },
  {
    code: EXIT_RUNTIME_STOPPED,
    name: "runtime_stopped",
    summary: "The query succeeded, but the local Runtime isn't running.",
  },
  {
    code: EXIT_PLAN_BLOCKED,
    name: "plan_blocked",
    summary:
      "The plan was generated, but the current platform or environment doesn't allow it to run; no changes were made.",
  },
];
