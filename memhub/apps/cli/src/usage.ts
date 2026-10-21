import {
  CAPABILITY_ALIASES,
  COMMANDS,
  type CommandDescriptor,
  type CommandOption,
} from "./command-catalog.js";
import { EXIT_CODES, EXIT_USAGE } from "./exit-codes.js";
import { CLI_VERSION, PRODUCT } from "./version.js";

const GLOBAL_OPTIONS: readonly CommandOption[] = [
  {
    flag: "--help, -h",
    summary:
      "Show help; can be combined with --json for a machine-readable catalog.",
  },
  { flag: "--version, -V", summary: "Show the CLI version." },
  {
    flag: "--json",
    summary: "Emit JSON on stdout for scripts and agents to consume.",
  },
];

export function renderGeneralHelp(): string {
  return [
    `MemHub CLI ${CLI_VERSION}`,
    "",
    `Usage: ${PRODUCT} <command> [options]`,
    "",
    "Commands:",
    ...COMMANDS.map((command) => `  ${pad(command.name)}${command.summary}`),
    "",
    "Global options:",
    ...GLOBAL_OPTIONS.map((option) => `  ${pad(option.flag)}${option.summary}`),
    "",
    "Exit codes:",
    ...EXIT_CODES.map(
      (exit) => `  ${exit.code}  ${pad(exit.name, 17)}${exit.summary}`,
    ),
    "",
    "stdout carries data only; progress and diagnostics go to stderr.",
    `Run ${PRODUCT} help <command> for a single command's arguments.`,
    "",
  ].join("\n");
}

export function renderCommandHelp(command: CommandDescriptor): string {
  const lines = [`Usage: ${command.usage}`, ""];
  if (command.alternative_usage) {
    lines.push(command.alternative_usage, "");
  }
  lines.push(command.summary, "");
  if (command.options.length > 0) {
    lines.push(
      "Options:",
      ...command.options.map(
        (option) => `  ${pad(option.flag, 34)}${option.summary}`,
      ),
      "",
    );
  }
  lines.push(
    "Exit codes:",
    ...EXIT_CODES.filter((exit) => command.exit_codes.includes(exit.code)).map(
      (exit) => `  ${exit.code}  ${pad(exit.name, 17)}${exit.summary}`,
    ),
    "",
  );
  return lines.join("\n");
}

export function renderCatalogDocument(command?: CommandDescriptor): string {
  const identity = {
    schema_version: 1,
    product: PRODUCT,
    cli_version: CLI_VERSION,
  };
  return `${JSON.stringify(
    command
      ? { ...identity, command }
      : {
          ...identity,
          commands: COMMANDS,
          capability_aliases: CAPABILITY_ALIASES,
          exit_codes: EXIT_CODES,
        },
  )}\n`;
}

export function renderVersion(): string {
  return `${PRODUCT} ${CLI_VERSION}\n`;
}

export function renderVersionDocument(): string {
  return `${JSON.stringify({
    schema_version: 1,
    product: PRODUCT,
    cli_version: CLI_VERSION,
  })}\n`;
}

export function renderUsageError(
  command: CommandDescriptor | undefined,
  message: string,
): string {
  const nextStep = usageNextStep(command);
  const lines = [`${PRODUCT}: ${message}`];
  if (command) lines.push(`Usage: ${command.usage}`);
  lines.push(`Next: ${nextStep}`, "");
  return lines.join("\n");
}

export function renderUsageErrorDocument(
  command: CommandDescriptor | undefined,
  message: string,
): string {
  return `${JSON.stringify({
    schema_version: 1,
    product: PRODUCT,
    status: "usage_error",
    command: command?.name ?? null,
    usage: command?.usage ?? null,
    exit_code: EXIT_USAGE,
    message,
    next_step: usageNextStep(command),
  })}\n`;
}

function usageNextStep(command: CommandDescriptor | undefined): string {
  return command
    ? `Run ${PRODUCT} help ${command.name} for this command's arguments.`
    : `Run ${PRODUCT} help to list available commands.`;
}

function pad(value: string, width = 21): string {
  return value.length >= width ? `${value} ` : value.padEnd(width);
}
