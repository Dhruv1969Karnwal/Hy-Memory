import { Copy, RefreshCw } from "lucide-react";
import { useState } from "react";
import { copyText } from "./copy-text";

type Material = { mcp: string; shell: string | null; prompt: string };
type Mode = keyof Material;

export function ConnectionFallback() {
  const [material, setMaterial] = useState<Material | null>(null);
  const [mode, setMode] = useState<Mode>("prompt");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const [notice, setNotice] = useState("");
  const available = !!window.oneMemory?.getConnectionMaterial;

  async function load() {
    const bridge = window.oneMemory;
    if (!bridge?.getConnectionMaterial || loading) return;
    setLoading(true);
    setError(false);
    try {
      setMaterial(await bridge.getConnectionMaterial());
    } catch {
      setError(true);
    } finally {
      setLoading(false);
    }
  }

  return (
    <details
      className="connection-fallback"
      onToggle={(event) => {
        if (event.currentTarget.open && !material && !loading && !error)
          void load();
      }}
    >
      <summary>Connect other tools</summary>
      {!available ? (
        <p>Local connection material is only available in the desktop app.</p>
      ) : (
        <>
          <p className="connection-boundary">
            Local MCP / command tools only. Pure web chat cannot connect to this
            machine directly.
          </p>
          <p>
            For tools in the list above, use the one-click install; this is only
            for tools not listed.
          </p>
          {loading && (
            <p role="status">Reading the local connection config...</p>
          )}
          {error && (
            <p role="alert">
              Could not read the connection config.
              <button
                type="button"
                className="quiet"
                onClick={() => void load()}
              >
                <RefreshCw size={15} aria-hidden="true" />
                Retry
              </button>
            </p>
          )}
          {material && (
            <>
              <div
                role="tablist"
                aria-label="Connection material format"
                className="connection-modes"
              >
                {(
                  [
                    ["prompt", "Connection prompt"],
                    ["mcp", "MCP config"],
                    ["shell", "Command"],
                  ] as const
                ).map(([id, label]) => (
                  <button
                    key={id}
                    type="button"
                    role="tab"
                    aria-selected={mode === id}
                    disabled={material[id] === null}
                    onClick={() => {
                      setMode(id);
                      setNotice("");
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <textarea
                aria-label="Local connection material"
                readOnly
                value={material[mode] ?? ""}
                spellCheck={false}
              />
              <p>
                It contains local paths but no memories or keys. The snippet
                runs this machine's MemHub connector; confirm before pasting,
                and keep MemHub running.
              </p>
              <button
                type="button"
                className="secondary"
                onClick={async () => {
                  try {
                    await copyText(material[mode] ?? "");
                    setNotice("Copied; no connection has been established.");
                  } catch {
                    setNotice(
                      "Copy failed; you can select the text above to copy.",
                    );
                  }
                }}
              >
                <Copy size={15} aria-hidden="true" />
                Copy
                {mode === "prompt"
                  ? " prompt"
                  : mode === "mcp"
                    ? " config"
                    : " command"}
              </button>
              {notice && <p role="status">{notice}</p>}
            </>
          )}
        </>
      )}
    </details>
  );
}
