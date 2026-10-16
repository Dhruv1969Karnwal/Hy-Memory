import path from "node:path";
import { pathToFileURL } from "node:url";

import { net, protocol } from "electron";

export const APP_ORIGIN = "one-memory://app";

export function resolveWebResource(
  webRoot: string,
  requestUrl: string,
): string | null {
  const url = new URL(requestUrl);
  if (url.protocol !== "one-memory:" || url.host !== "app") return null;
  const pathname = decodeURIComponent(url.pathname);
  const requested = pathname === "/" ? "index.html" : pathname.slice(1);
  const resolvedRoot = path.resolve(webRoot);
  const resolved = path.resolve(resolvedRoot, requested);
  const relative = path.relative(resolvedRoot, resolved);
  if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) {
    return null;
  }
  return resolved;
}

export function installAppProtocol(webRoot: string): void {
  protocol.handle("one-memory", (request) => {
    const resource = resolveWebResource(webRoot, request.url);
    if (!resource) {
      return new Response("Not found", { status: 404 });
    }
    return net.fetch(pathToFileURL(resource).toString());
  });
}
