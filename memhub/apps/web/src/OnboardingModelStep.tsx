import type { ProviderKind } from "@one-memory/contracts";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";
import { OptionMenu } from "./OptionMenu";
import { R6ColdStartFrame } from "./R6ColdStartFrame";
import { R6Button } from "./R6Controls";
import "./onboarding-model-r6.css";

/** Strings preserve partial numeric edits. The owner validates and maps to API payloads. */
export interface OnboardingModelDraft {
  provider_id: string;
  name: string;
  provider_kind: ProviderKind;
  model: string;
  base_url: string;
  api_key: string;
  disable_thinking: boolean;
  embedding_mode: "local" | "custom";
  embedding_provider_kind: "openai_compatible" | "ollama";
  embedding_model: string;
  embedding_base_url: string;
  timeout_seconds: string;
  max_concurrency: string;
  context_window: string;
}
export interface OnboardingModelProvider {
  id: string;
  label: string;
  keyRequired: boolean;
  supported: boolean;
}
type Action = "test" | "save-draft" | "save-and-enable";
export interface OnboardingModelStepProps {
  draft: OnboardingModelDraft;
  providers: readonly OnboardingModelProvider[];
  advancedOpen: boolean;
  onAdvancedChange(open: boolean): void;
  onDraftChange(patch: Partial<OnboardingModelDraft>): void;
  phase: "idle" | "testing" | "saving" | "enabling";
  capability: { saveAndEnable: boolean; reason?: string };
  canTest: boolean;
  canSaveDraft: boolean;
  /** Owner validation/current revision gate; absence fails closed. */
  canSaveAndEnable?: boolean;
  enableActionLabel?: string;
  testActionLabel?: string;
  inputsLocked?: boolean;
  fieldErrors?: Partial<Record<keyof OnboardingModelDraft, string>>;
  error?: string | null;
  probe?: { state: "passed" | "failed" | "stale"; message: string } | null;
  profile?: {
    profile_id: string;
    revision: number;
    has_configured_api_key: boolean;
    credential_state?: "available" | "missing" | "unavailable";
  } | null;
  sourceNames: readonly string[];
  importedConversationCount?: number;
  connectionSummary?: string;
  generationRecipient: string;
  embeddingRecipient: string;
  onTest(): Promise<void>;
  onSaveAndEnable(): Promise<void>;
  onSaveDraft(): Promise<void>;
  onRecoverCreation?(): Promise<void>;
  onDiscardMissingCreation?(): Promise<void>;
  onBack(): void;
  onSkip?(): void;
  /** Owner must preserve receipts/unknown results and supply a safe display error. */
  onActionError?(action: Action): void;
}

export function OnboardingModelStep(props: OnboardingModelStepProps) {
  const { draft, advancedOpen, onDraftChange, fieldErrors = {}, phase } = props;
  const id = useId();
  const fields = useRef<HTMLElement>(null);
  const advancedToggle = useRef<HTMLButtonElement>(null);
  const restoreToggleFocus = useRef(false);
  const flight = useRef(false);
  const alive = useRef(true);
  const [pending, setPending] = useState<Action | null>(null);
  const [callbackFailed, setCallbackFailed] = useState(false);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  const locked = phase !== "idle" || pending !== null;
  const fieldsLocked = locked || props.inputsLocked === true;
  // biome-ignore lint/correctness/useExhaustiveDependencies: Restore only after the controlled mode change has rendered its replacement toggle.
  useEffect(() => {
    if (!restoreToggleFocus.current) return;
    restoreToggleFocus.current = false;
    if (fields.current) fields.current.scrollTop = 0;
    advancedToggle.current?.focus({ preventScroll: true });
  }, [advancedOpen]);
  function changeAdvanced(open: boolean) {
    restoreToggleFocus.current = true;
    props.onAdvancedChange(open);
  }
  const provider = props.providers.find(
    (item) => item.id === draft.provider_id,
  );
  const canEnable =
    (props.probe?.state === "passed" ||
      props.enableActionLabel === "Verify submission status") &&
    props.capability.saveAndEnable &&
    props.canSaveAndEnable === true &&
    !!props.generationRecipient.trim() &&
    !!props.embeddingRecipient.trim() &&
    provider?.supported;
  const sourceSummary =
    props.sourceNames.length <= 3
      ? props.sourceNames.join(" · ")
      : `${props.sourceNames.slice(0, 3).join(" · ")} and ${props.sourceNames.length} apps in total`;
  async function run(action: Action) {
    if (flight.current || locked) return;
    if (action === "save-and-enable" && !canEnable) return;
    if (action === "test" && (!props.canTest || !provider?.supported)) return;
    if (action === "save-draft" && !props.canSaveDraft) return;
    flight.current = true;
    setPending(action);
    setCallbackFailed(false);
    try {
      await (action === "test"
        ? props.onTest()
        : action === "save-draft"
          ? props.onSaveDraft()
          : props.onSaveAndEnable());
    } catch {
      if (alive.current) {
        setCallbackFailed(true);
        props.onActionError?.(action);
      }
    } finally {
      flight.current = false;
      if (alive.current) setPending(null);
    }
  }
  function field(
    key: keyof OnboardingModelDraft,
    label: string,
    options: {
      type?: string;
      min?: number;
      max?: number;
      maxLength?: number;
      required?: boolean;
      placeholder?: string;
    } = {},
  ) {
    const error = fieldErrors[key];
    return (
      <label className="onboarding-model-field" htmlFor={`${id}-${key}`}>
        <span>{label}</span>
        <input
          id={`${id}-${key}`}
          name={key}
          aria-label={label}
          type={options.type ?? "text"}
          autoComplete="off"
          spellCheck={false}
          value={String(draft[key])}
          disabled={fieldsLocked}
          aria-invalid={!!error}
          aria-describedby={error ? `${id}-${key}-error` : undefined}
          min={options.min}
          max={options.max}
          maxLength={options.maxLength}
          required={options.required}
          placeholder={options.placeholder}
          onChange={(event) => {
            if (!flight.current && !fieldsLocked)
              onDraftChange({ [key]: event.target.value });
          }}
        />
        {error && (
          <small id={`${id}-${key}-error`} className="onboarding-model-error">
            {error}
          </small>
        )}
      </label>
    );
  }
  const displayedPhase =
    phase !== "idle"
      ? phase
      : pending === "test"
        ? "testing"
        : pending === "save-draft"
          ? "saving"
          : pending
            ? "enabling"
            : "idle";
  const connectionFailed = props.probe?.state === "failed";
  const pageTitle = advancedOpen
    ? "Configuration details, adjust as needed"
    : connectionFailed
      ? "Connection failed; the configuration is kept"
      : "Pick the model and start distilling memories";
  const connectionReceipt = props.connectionSummary?.includes("enabled")
    ? "Connection settings saved\nSuccessful items are kept and not repeated."
    : props.connectionSummary === "Continuous sync · One-click connect"
      ? "Selected: Continuous sync · One-click connect"
      : props.connectionSummary;
  return (
    <div
      className="onboarding-model-r6"
      data-integration={
        props.capability.saveAndEnable
          ? "capability-provided"
          : "not-integrated"
      }
    >
      <R6ColdStartFrame
        step={3}
        eyebrow="Distill memories · Use the model you chose"
        title={pageTitle}
        footer={
          <>
            <p className="onboarding-model-footer-note">
              {!props.capability.saveAndEnable
                ? "A model cannot be enabled right now; your entries are kept."
                : props.probe?.state === "passed"
                  ? "The model connection test passed; you can enable it and start distilling memories."
                  : "Test the model connection first; enable it once the test passes."}
            </p>
            <div className="onboarding-model-footer-actions">
              <R6Button
                disabled={locked}
                onClick={() => {
                  if (!flight.current && !locked)
                    (props.onSkip ?? props.onBack)();
                }}
              >
                Skip and go to the app
              </R6Button>
              <R6Button
                disabled={locked || !props.canTest}
                onClick={() => void run("test")}
              >
                {displayedPhase === "testing"
                  ? "Testing the model..."
                  : (props.testActionLabel ?? "Test the model connection")}
              </R6Button>
              <R6Button
                variant="primary"
                disabled={locked || !canEnable}
                onClick={() => void run("save-and-enable")}
              >
                {displayedPhase === "enabling"
                  ? "Enabling..."
                  : (props.enableActionLabel ?? "Confirm and enable")}
              </R6Button>
            </div>
          </>
        }
      >
        <div className="onboarding-model-layout">
          <section
            className="onboarding-model-form"
            aria-label="Model configuration"
            data-advanced={advancedOpen}
          >
            <section
              ref={fields}
              id={`${id}-fields`}
              className="onboarding-model-fields"
              aria-label="Model parameters"
              // biome-ignore lint/a11y/noNoninteractiveTabindex: The parameter pane needs keyboard scrolling.
              tabIndex={0}
            >
              {!advancedOpen ? (
                <>
                  <p className="onboarding-model-intro">
                    Pick a provider and enter the API Key. Check the existing
                    configuration before continuing.
                  </p>
                  <label
                    className="onboarding-model-field onboarding-model-provider"
                    htmlFor={`${id}-provider`}
                  >
                    <span id={`${id}-provider-label`}>Model provider</span>
                    <OptionMenu
                      id={`${id}-provider`}
                      labelledBy={`${id}-provider-label`}
                      value={draft.provider_id}
                      disabled={fieldsLocked}
                      options={[
                        ...(!provider
                          ? [{ value: "", label: "Choose a provider" }]
                          : []),
                        ...props.providers.map((item) => ({
                          value: item.id,
                          label: item.label,
                          disabled: !item.supported,
                        })),
                      ]}
                      onChange={(value) => {
                        if (!flight.current)
                          onDraftChange({ provider_id: value });
                      }}
                    />
                  </label>
                  <div className="onboarding-model-key">
                    {field("api_key", "API Key", {
                      type: "password",
                      required:
                        provider?.keyRequired &&
                        !props.profile?.has_configured_api_key,
                      placeholder: props.profile?.has_configured_api_key
                        ? "Saved securely; leave blank to keep it"
                        : "Enter the API Key from your provider",
                    })}
                  </div>
                  <p
                    className="onboarding-model-key-note"
                    data-state={connectionFailed ? "failed" : "idle"}
                  >
                    {connectionFailed
                      ? "The connection failed. Check the API Key, model and endpoint and retry. Your configuration is kept."
                      : props.profile?.credential_state === "unavailable"
                        ? "MemHub needs permission to access the keychain. Choose 'Always allow' when the system asks."
                        : props.profile?.has_configured_api_key
                          ? "Leave blank to keep the saved key; a new key needs a new test."
                          : "The key is kept in the system secure store and never written into conversations or logs."}
                  </p>
                  <button
                    ref={advancedToggle}
                    type="button"
                    className="onboarding-model-toggle"
                    aria-expanded={false}
                    aria-controls={`${id}-fields`}
                    disabled={fieldsLocked}
                    onClick={() => changeAdvanced(true)}
                  >
                    Configuration details and advanced settings{" "}
                    <ChevronDown size={16} aria-hidden="true" />
                  </button>
                  <p className="onboarding-model-profile">
                    {provider?.label ?? "Provider not chosen"} ·{" "}
                    {draft.model || "Model not filled in"}
                    <br />
                    {safeEndpoint(draft.base_url)} ·{" "}
                    {draft.embedding_mode === "local"
                      ? "Local embedding model"
                      : draft.embedding_model ||
                        "Embedding model not filled in"}{" "}
                    ·{" "}
                    {draft.disable_thinking
                      ? "Thinking off"
                      : "Thinking not off"}
                  </p>
                </>
              ) : (
                <>
                  <div className="onboarding-model-advanced-head">
                    <p>
                      The presets are filled in; only custom services or special
                      needs require changes.
                    </p>
                    <button
                      ref={advancedToggle}
                      type="button"
                      className="onboarding-model-toggle"
                      aria-expanded={true}
                      aria-controls={`${id}-fields`}
                      disabled={fieldsLocked}
                      onClick={() => changeAdvanced(false)}
                    >
                      Back to provider selection{" "}
                      <ChevronUp size={16} aria-hidden="true" />
                    </button>
                  </div>
                  <div className="onboarding-model-advanced-grid">
                    {field("name", "Configuration name", { maxLength: 128 })}
                    {field("model", "Model name", {
                      required: true,
                      maxLength: 256,
                    })}
                    <div className="onboarding-model-wide">
                      {field("base_url", "Base URL", {
                        type: "url",
                        required: true,
                      })}
                    </div>
                    <div className="onboarding-model-wide onboarding-model-key">
                      {field("api_key", "API Key", {
                        type: "password",
                        required:
                          provider?.keyRequired &&
                          !props.profile?.has_configured_api_key,
                        placeholder: props.profile?.has_configured_api_key
                          ? "Saved securely; leave blank to keep it"
                          : "Enter the API Key from your provider",
                      })}
                      <p
                        className="onboarding-model-key-note"
                        data-state={connectionFailed ? "failed" : "idle"}
                      >
                        {props.profile?.has_configured_api_key
                          ? "Leave blank to keep the saved key; a new key needs a new test."
                          : "The key is kept in the system secure store and never written into conversations or logs."}
                      </p>
                    </div>
                    <label className="onboarding-model-field">
                      <span>Protocol</span>
                      <select
                        aria-label="Protocol"
                        value={draft.provider_kind}
                        disabled={fieldsLocked}
                        onChange={(e) => {
                          if (!flight.current)
                            onDraftChange({
                              provider_kind: e.target.value as ProviderKind,
                            });
                        }}
                      >
                        <option value="openai_compatible">
                          OpenAI-compatible
                        </option>
                        <option value="ollama">Ollama</option>
                        <option value="anthropic" disabled>
                          Anthropic (not supported yet)
                        </option>
                        <option value="hy_default" disabled>
                          HY hosted (currently unavailable)
                        </option>
                      </select>
                    </label>
                    <label className="onboarding-model-field">
                      <span>Embedding model</span>
                      <select
                        aria-label="Embedding model"
                        value={draft.embedding_mode}
                        disabled={fieldsLocked}
                        onChange={(e) => {
                          if (!flight.current)
                            onDraftChange({
                              embedding_mode: e.target.value as
                                | "local"
                                | "custom",
                            });
                        }}
                      >
                        <option value="local">Local ONNX · built-in</option>
                        <option value="custom">Custom embedding</option>
                      </select>
                    </label>
                    <div className="onboarding-model-numbers">
                      {field("timeout_seconds", "Timeout (seconds)", {
                        type: "number",
                        min: 1,
                        max: 600,
                      })}
                      {field("max_concurrency", "Max concurrency", {
                        type: "number",
                        min: 1,
                        max: 32,
                      })}
                      {field("context_window", "Context window", {
                        type: "number",
                        min: 1024,
                        max: 10000000,
                        placeholder: "Automatic / optional",
                      })}
                    </div>
                    <label className="onboarding-model-thinking">
                      <input
                        type="checkbox"
                        checked={draft.disable_thinking}
                        disabled={fieldsLocked}
                        onChange={(e) => {
                          if (!flight.current)
                            onDraftChange({
                              disable_thinking: e.target.checked,
                            });
                        }}
                      />
                      Turn off thinking mode
                    </label>
                    <p className="onboarding-model-vector-note">
                      A custom embedding configuration can be expanded; a remote
                      embedding service must respect the credential boundary.
                      <br />
                      After switching provider or model you have to test and
                      enable again.
                    </p>
                    {draft.embedding_mode === "custom" && (
                      <>
                        {field("embedding_model", "Embedding model name", {
                          required: true,
                          maxLength: 256,
                        })}
                        <label className="onboarding-model-field">
                          <span>Embedding protocol</span>
                          <select
                            aria-label="Embedding protocol"
                            value={draft.embedding_provider_kind}
                            disabled={fieldsLocked}
                            onChange={(e) => {
                              if (!flight.current)
                                onDraftChange({
                                  embedding_provider_kind: e.target.value as
                                    | "ollama"
                                    | "openai_compatible",
                                });
                            }}
                          >
                            <option value="openai_compatible">
                              Same-origin OpenAI-compatible (shared Key)
                            </option>
                            <option value="ollama">Local Ollama</option>
                          </select>
                        </label>
                        <div className="onboarding-model-wide">
                          {field("embedding_base_url", "Embedding Base URL", {
                            type: "url",
                            required: true,
                          })}
                          <small>
                            The remote embedding address must share the
                            generation service origin; local addresses are still
                            validated.
                          </small>
                        </div>
                      </>
                    )}
                  </div>
                </>
              )}
            </section>
            <div className="onboarding-model-recovery-actions">
              {props.onRecoverCreation && (
                <R6Button
                  disabled={locked}
                  onClick={() => void props.onRecoverCreation?.()}
                >
                  Verify the creation result
                </R6Button>
              )}
              {props.onDiscardMissingCreation && (
                <R6Button
                  variant="secondary"
                  disabled={locked}
                  onClick={() => void props.onDiscardMissingCreation?.()}
                >
                  Drop the old record and reconfigure
                </R6Button>
              )}
            </div>
            <div className="onboarding-model-feedback" tabIndex={-1}>
              {(displayedPhase === "testing" ||
                displayedPhase === "enabling") && (
                <p role="status" aria-label="Model submission status">
                  {displayedPhase === "testing"
                    ? "Testing the connection..."
                    : "Submitting the save and enable; waiting for the receipt..."}
                </p>
              )}
              {props.probe && (
                <p
                  className="onboarding-model-probe"
                  data-state={props.probe.state}
                  role="status"
                >
                  {props.probe.message}
                </p>
              )}
              {(props.error || callbackFailed) && (
                <p className="onboarding-model-error" role="alert">
                  {props.error ||
                    "The operation result is unconfirmed; your entries are kept. Check and retry."}
                </p>
              )}
            </div>
          </section>
          <aside
            className="onboarding-model-summary"
            aria-label="This distillation"
            data-full-height={advancedOpen || connectionFailed}
            // biome-ignore lint/a11y/noNoninteractiveTabindex: Keyboard users must be able to scroll the full permission disclosure.
            tabIndex={0}
          >
            <div className="onboarding-model-summary-heading">
              <p>This distillation</p>
              <h2>Memories start here</h2>
            </div>
            <div className="onboarding-model-connections">
              <p>Conversation sources</p>
              <strong>{sourceSummary || "No sources selected yet"}</strong>
              <small>
                Imported {props.importedConversationCount ?? 0} conversations;
                the source switches keep controlling what syncs to this machine
                later.
              </small>
            </div>
            <div className="onboarding-model-summary-status">
              {connectionReceipt && (
                <small className="onboarding-model-connection-receipt">
                  {connectionReceipt}
                </small>
              )}
              <small>
                Processing service: {provider?.label ?? "Not chosen"}
                <br />
                The relevant conversations are sent to the selected provider.
              </small>
              <small>
                Testing the connection only sends test content. Once enabled,
                this model organizes imported and later synced conversations.
              </small>
            </div>
            {!props.capability.saveAndEnable && (
              <p role="status">
                {props.capability.reason ||
                  "A model cannot be enabled here right now."}
              </p>
            )}
          </aside>
        </div>
      </R6ColdStartFrame>
    </div>
  );
}

function safeEndpoint(value: string): string {
  try {
    const url = new URL(value);
    if (
      !["http:", "https:"].includes(url.protocol) ||
      url.username ||
      url.password ||
      url.search ||
      url.hash
    )
      return "Endpoint pending verification";
    return `${url.origin}${url.pathname.replace(/\/$/, "")}`;
  } catch {
    return "Endpoint pending verification";
  }
}
