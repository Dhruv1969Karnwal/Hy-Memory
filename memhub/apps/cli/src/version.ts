declare const __ONE_MEMORY_VERSION__: string;
declare const __ONE_MEMORY_BUILD_NUMBER__: string;

const embeddedVersion =
  typeof __ONE_MEMORY_VERSION__ === "string"
    ? __ONE_MEMORY_VERSION__.trim()
    : "";
const embeddedBuildNumber =
  typeof __ONE_MEMORY_BUILD_NUMBER__ === "string"
    ? __ONE_MEMORY_BUILD_NUMBER__.trim()
    : "";

export const CLI_VERSION =
  /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$/.test(
    embeddedVersion,
  )
    ? embeddedVersion
    : "development";

export const CLI_BUILD_NUMBER = /^\d+$/.test(embeddedBuildNumber)
  ? Number(embeddedBuildNumber)
  : null;

export const PRODUCT = "memhub";
