import type { ModelProfile } from "@one-memory/contracts";
import { Download, RefreshCw } from "lucide-react";
import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { getModelProfiles } from "./api";
import { ConnectionFallback } from "./ConnectionFallback";
import { exportDiagnosticsFile } from "./diagnostics-export";
import { ModelSettings, type ModelSettingsActions } from "./ModelSettings";
import { R6Button, R6Switch } from "./R6Controls";
import {
  refreshLoginPreference,
  setLoginPreference,
  useLoginPreference,
} from "./runtime-state";
import "./r6-settings.css";

type Group = "model" | "connections" | "help";

const labels: Record<Group, string> = {
  model: "Model",
  connections: "Connect other apps",
  help: "Help and diagnostics",
};
const primaryGroups: Group[] = ["model", "connections"];

function safeEndpoint(value: string | null) {
  if (!value) return "Local / managed endpoint";
  try {
    const endpoint = new URL(value);
    return `${endpoint.origin}${endpoint.pathname.replace(/\/$/, "")}`;
  } catch {
    return "Endpoint pending verification";
  }
}

export function R6Settings({
  active = true,
  onDiagnostics: _onDiagnostics,
  showDiagnostics = import.meta.env.VITE_ENABLE_DIAGNOSTICS !== "false",
}: {
  onDiagnostics(): void;
  active?: boolean;
  /** Shown by default; a release build may still opt out with VITE_ENABLE_DIAGNOSTICS=false. */
  showDiagnostics?: boolean;
}) {
  const id = useId();
  const groups = showDiagnostics
    ? [...primaryGroups, "help" as const]
    : primaryGroups;
  const [group, setGroup] = useState<Group>("model");
  const [profiles, setProfiles] = useState<ModelProfile[] | null>(null);
  const [modelError, setModelError] = useState(false);
  const [configurationRefusal, setConfigurationRefusal] = useState(false);
  const [modelVersion, setModelVersion] = useState(0);
  const [configurationOpen, setConfigurationOpen] = useState(false);
  const [quitOpen, setQuitOpen] = useState(false);
  const [quitting, setQuitting] = useState(false);
  const [quitError, setQuitError] = useState("");
  const [exportingLogs, setExportingLogs] = useState(false);
  const [exportMessage, setExportMessage] = useState("");
  const modelActions = useRef<ModelSettingsActions>(null);
  const quitPanel = useRef<HTMLElement>(null);
  const quitOpener = useRef<HTMLElement | null>(null);
  const currentModel = profiles?.find((profile) => profile.is_active) ?? null;

  useLayoutEffect(() => {
    if (!active) return;
    void group;
    const content = document.getElementById("main-content");
    if (content) content.scrollTop = 0;
  }, [active, group]);

  async function handleExportLogs() {
    setExportingLogs(true);
    setExportMessage("");
    try {
      const result = await exportDiagnosticsFile();
      setExportMessage(result.message);
    } finally {
      setExportingLogs(false);
    }
  }

  const readModels = useCallback(async () => {
    setModelError(false);
    try {
      setProfiles(await getModelProfiles());
      setConfigurationRefusal(false);
    } catch {
      setModelError(true);
    }
  }, []);

  // biome-ignore lint/correctness/useExhaustiveDependencies: modelVersion is the explicit refresh trigger.
  useEffect(() => {
    if (!active) return;
    if (group !== "model") return;
    void readModels();
  }, [active, group, modelVersion, readModels]);

  const login = useLoginPreference() ?? null;
  const [dataReady, setDataReady] = useState(login !== null);
  const [loginPending, setLoginPending] = useState(false);
  const [dataError, setDataError] = useState("");
  const [readError, setReadError] = useState("");
  const [loginReadFailed, setLoginReadFailed] = useState(false);

  const readData = useCallback(async () => {
    try {
      await refreshLoginPreference();
      setReadError("");
      setLoginReadFailed(false);
    } catch {
      setLoginReadFailed(true);
      setReadError(
        "Some runtime state cannot be read right now; the last state that was read is shown. Try again.",
      );
    }
    setDataReady(true);
  }, []);

  useEffect(() => {
    if (!active) return;
    void readData();
  }, [active, readData]);

  async function changeLogin(enabled: boolean) {
    if (!window.oneMemory?.setLoginPreference) {
      setDataError(
        "This install cannot set the start-at-login background service yet.",
      );
      return;
    }
    setLoginPending(true);
    setDataError("");
    try {
      const nextLogin = await window.oneMemory.setLoginPreference({
        openAtLogin: enabled,
        userConfirmed: true,
      });
      setLoginPreference(nextLogin);
    } catch {
      setDataError("The start-at-login setting was not confirmed; try again.");
      await readData();
    } finally {
      setLoginPending(false);
    }
  }

  function openConfiguration() {
    setConfigurationRefusal(false);
    const opened = currentModel
      ? modelActions.current?.editProfile(currentModel)
      : modelActions.current?.add();
    if (opened) setConfigurationOpen(true);
    else setConfigurationRefusal(true);
  }

  async function quitApplication() {
    if (quitting) return;
    if (!window.oneMemory?.requestQuit) {
      setQuitError(
        "This install cannot quit the interface; quit from the menu bar.",
      );
      return;
    }
    setQuitting(true);
    setQuitError("");
    try {
      await window.oneMemory.requestQuit(true);
    } catch {
      setQuitError("Quitting the interface was not confirmed; try again.");
    } finally {
      setQuitting(false);
    }
  }

  useEffect(() => {
    if (!quitOpen) return;
    const before = quitOpener.current;
    quitPanel.current
      ?.querySelector<HTMLButtonElement>("button:not(:disabled)")
      ?.focus();
    return () => {
      if (before?.isConnected) before.focus();
    };
  }, [quitOpen]);

  function moveGroupFocus(event: React.KeyboardEvent<HTMLButtonElement>) {
    const focused = (event.currentTarget.id.split("-nav-").at(-1) ??
      group) as Group;
    const current = groups.indexOf(focused);
    let next: number | null = null;
    if (event.key === "ArrowDown") next = (current + 1) % groups.length;
    else if (event.key === "ArrowUp")
      next = (current - 1 + groups.length) % groups.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = groups.length - 1;
    if (next === null) return;
    event.preventDefault();
    document.getElementById(`${id}-nav-${groups[next]}`)?.focus();
  }

  return (
    <section className="r6-settings" aria-label="Settings">
      <h1>Settings</h1>
      <div className="r6-settings-layout">
        <nav aria-label="Settings groups">
          {groups.map((key) => (
            <button
              type="button"
              key={key}
              id={`${id}-nav-${key}`}
              aria-controls={`${id}-panel-${key}`}
              aria-current={group === key ? "page" : undefined}
              onKeyDown={moveGroupFocus}
              onClick={() => setGroup(key)}
            >
              {labels[key]}
            </button>
          ))}
        </nav>
        <div className="r6-settings-content">
          <section
            id={`${id}-panel-model`}
            aria-labelledby={`${id}-nav-model`}
            hidden={group !== "model"}
            className="r6-settings-model"
          >
            <h2>Memory model</h2>
            <div className="r6-model-overview">
              <p className="r6-model-caption">Current model</p>
              <h3 className="r6-current-model">
                {currentModel?.name ??
                  (profiles === null
                    ? "Reading the model..."
                    : modelError
                      ? "Model state unavailable"
                      : "No model configured")}
              </h3>
              {currentModel && (
                <p className="r6-model-state">
                  {currentModel.model} · {safeEndpoint(currentModel.base_url)}
                </p>
              )}
              <div className="r6-model-actions">
                <R6Button
                  variant="primary"
                  disabled={profiles === null}
                  onClick={openConfiguration}
                >
                  {currentModel ? "Change model" : "Configure model"}
                </R6Button>
                <R6Button
                  variant="quiet"
                  aria-label="Refresh model status"
                  title="Refresh model status"
                  onClick={() => setModelVersion((value) => value + 1)}
                >
                  <RefreshCw size={16} aria-hidden="true" />
                </R6Button>
              </div>
              {modelError && (
                <p role="alert" className="r6-settings-error">
                  {profiles === null
                    ? "The model status could not be read; try again."
                    : "The model status did not refresh; the last confirmed model status is still shown. Try again."}
                </p>
              )}
              {configurationRefusal && (
                <p role="alert" className="r6-settings-error">
                  Model configuration cannot open right now: the earlier
                  creation or test is still in flight, or local storage is
                  unavailable. Wait for the current operation to finish and try
                  again; the existing configuration and source text are not
                  changed.
                </p>
              )}
            </div>
            <section
              className="r6-model-owner"
              hidden={!configurationOpen}
              aria-label="Model configuration form"
            >
              <ModelSettings
                compactSingle
                controlsRef={modelActions}
                onProfilesChanged={() => setModelVersion((value) => value + 1)}
                onSetupComplete={() => setConfigurationOpen(false)}
              />
            </section>
          </section>
          <section
            id={`${id}-panel-connections`}
            aria-labelledby={`${id}-nav-connections`}
            hidden={group !== "connections"}
            className="r6-settings-connections"
          >
            <h2>Connect other apps</h2>
            <ConnectionFallback />
          </section>
          {showDiagnostics && (
            <section
              id={`${id}-panel-help`}
              aria-labelledby={`${id}-nav-help`}
              hidden={group !== "help"}
              className="r6-settings-help"
            >
              <h2>Help and diagnostics</h2>
              <div className="r6-setting-row">
                <div>
                  <h3>Start the background service at login</h3>
                  <p>
                    Syncs, organizes and serves memories without opening the
                    interface.
                  </p>
                </div>
                <R6Switch
                  label="Start the background service at login"
                  checked={login?.enabled === true}
                  disabled={
                    !dataReady ||
                    loginPending ||
                    loginReadFailed ||
                    login?.supported !== true
                  }
                  onCheckedChange={(enabled) => void changeLogin(enabled)}
                />
              </div>
              {(dataError || readError) && (
                <p role="alert" className="r6-settings-error">
                  {dataError || readError}
                </p>
              )}
              <div className="r6-setting-row">
                <div>
                  <h3>Export logs</h3>
                  <p>
                    {exportMessage ||
                      "Bundles the local runtime logs into one file to send to development."}
                  </p>
                </div>
                <R6Button
                  variant="quiet"
                  pending={exportingLogs}
                  onClick={() => void handleExportLogs()}
                >
                  <Download size={16} aria-hidden="true" />
                  Export logs
                </R6Button>
              </div>
            </section>
          )}
        </div>
      </div>
      {quitOpen && (
        <div className="r6-settings-dialog-layer" role="presentation">
          <section
            className="r6-settings-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby={`${id}-quit-title`}
            ref={quitPanel}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                event.preventDefault();
                if (!quitting) setQuitOpen(false);
                return;
              }
              if (event.key !== "Tab" || !quitPanel.current) return;
              const controls = [
                ...quitPanel.current.querySelectorAll<HTMLElement>(
                  "button:not(:disabled)",
                ),
              ];
              const first = controls[0];
              const last = controls.at(-1);
              if (!first || !last) return;
              if (event.shiftKey && document.activeElement === first) {
                event.preventDefault();
                last.focus();
              } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault();
                first.focus();
              }
            }}
          >
            <h2 id={`${id}-quit-title`}>Quit the One Memory interface?</h2>
            <p>
              The background service keeps syncing, organizing and serving
              memories.
            </p>
            {quitError && (
              <p role="alert" className="r6-settings-error">
                {quitError}
              </p>
            )}
            <footer>
              <R6Button disabled={quitting} onClick={() => setQuitOpen(false)}>
                Cancel
              </R6Button>
              <R6Button
                variant="primary"
                pending={quitting}
                onClick={() => void quitApplication()}
              >
                Quit the interface
              </R6Button>
            </footer>
          </section>
        </div>
      )}
    </section>
  );
}
