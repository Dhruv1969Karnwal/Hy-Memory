import type {
  ModelProbeResult,
  ModelProfile,
  ModelProfileInput,
  ModelProfilePatch,
  ProbeCode,
  ProviderKind,
} from "@one-memory/contracts";
import { Check, Pencil, Plus, RefreshCw } from "lucide-react";
import {
  type FormEvent,
  type Ref,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { OptionMenu } from "./OptionMenu";
import { PROVIDER_PRESETS } from "./provider-presets";
import { useModelProfiles } from "./use-model-profiles";
import "./model-setup-next.css";

/** The vectoriser that ships with the app. Not a choice, so not a setting. */
const BUILTIN_EMBEDDING_KIND = "local_onnx";

/**
 * The name `seed_preset_profile` gives the profile an internal build creates.
 * Matched by name because that is all the contract carries: an `is_preset`
 * field would put a build-time distinction into the public API, and the API
 * contract is not something this change is allowed to move.
 */
const PRESET_PROFILE_NAME = "Preconfigured model service";

/**
 * A configured distribution may point to public provider documentation.
 */
function applyUrl(baseUrl: string | null): string | null {
  if (!baseUrl) return null;
  try {
    const url = new URL(baseUrl);
    return `${url.protocol}//${url.hostname.replace(/^api\./, "")}`;
  } catch {
    return null;
  }
}

function presetKeyLabel(): string {
  return "Paste API Key";
}

const PROVIDERS: Array<{
  kind: ProviderKind;
  label: string;
  baseUrl: string;
  keyRequired: boolean;
  // The backend ModelProfileStore.activate() currently only allows openai_compatible and ollama;
  // the rest throw model_provider_not_supported. We flag them here and disable them in the dropdown,
  // so users do not fill the whole form, succeed at creation, and hit the wall only at activation.
  activatable: boolean;
}> = [
  {
    kind: "openai_compatible",
    label: "OpenAI-compatible",
    baseUrl: "https://api.openai.com/v1",
    keyRequired: true,
    activatable: true,
  },
  {
    kind: "anthropic",
    label: "Anthropic Messages (not supported yet)",
    baseUrl: "https://api.anthropic.com",
    keyRequired: true,
    activatable: false,
  },
  {
    kind: "ollama",
    label: "Ollama (local)",
    baseUrl: "http://127.0.0.1:11434",
    keyRequired: false,
    activatable: true,
  },
  {
    kind: "hy_default",
    label: "HY Memory hosted model (currently unavailable)",
    baseUrl: "",
    keyRequired: false,
    activatable: false,
  },
];

type FieldName =
  | "model"
  | "base-url"
  | "api-key"
  | "embedding-model"
  | "embedding-base-url";

function keyError(value: string, required: boolean): string | undefined {
  if (!value.trim()) return required ? "Enter an API Key." : undefined;
  if (/^(?:[a-z][a-z\d+.-]*:\/\/|www\.|\/\/)/i.test(value.trim()))
    return "This needs an API Key, not a URL. Enter the key provided by the provider.";
  return undefined;
}

const FIELD_PROBE_TEXT: Partial<Record<ProbeCode, string>> = {
  unauthorized:
    "The provider rejected this API Key (401). Make sure the key belongs to the service address you entered.",
  model_not_found:
    "The provider has no such model name. Check it against the provider's model list.",
  unreachable:
    "Cannot reach the endpoint. Check the network and the service address.",
};

function probeFieldErrors(
  result: ModelProbeResult | null | undefined,
): Partial<Record<FieldName, string>> {
  if (!result) return {};
  const stages = new Map(result.stages.map((stage) => [stage.stage, stage]));
  const errors: Partial<Record<FieldName, string>> = {};
  const text = (stage: string) => {
    const stageResult = stages.get(stage);
    return stageResult && stageResult.code !== "ok"
      ? (FIELD_PROBE_TEXT[stageResult.code] ?? undefined)
      : undefined;
  };
  if (!result.capabilities.endpoint_reachable) {
    errors["base-url"] = text("endpoint") ?? FIELD_PROBE_TEXT.unreachable;
  }
  if (!result.capabilities.model_available) {
    if (stages.get("model")?.code === "unauthorized") {
      errors["api-key"] = FIELD_PROBE_TEXT.unauthorized;
    } else {
      errors.model = text("model") ?? FIELD_PROBE_TEXT.model_not_found;
    }
  }
  if (!result.capabilities.embedding_available) {
    errors["embedding-model"] ??=
      text("embedding") ?? FIELD_PROBE_TEXT.embedding_unavailable;
  }
  return errors;
}

function endpointKey(value: string): string {
  return value.trim().replace(/\/+$/, "").toLowerCase();
}

function presetForProfile(profile: {
  provider_kind: string;
  base_url: string | null;
}): string {
  return (
    PROVIDER_PRESETS.find(
      (item) =>
        item.provider === profile.provider_kind &&
        endpointKey(item.baseUrl) === endpointKey(profile.base_url ?? ""),
    )?.id ?? "custom"
  );
}

function validEndpoint(value: string): boolean {
  try {
    const url = new URL(value);
    return (
      ["http:", "https:"].includes(url.protocol) &&
      !!url.hostname &&
      !url.username &&
      !url.password
    );
  } catch {
    return false;
  }
}

function ready(
  profile: ModelProfile,
  probe: ModelProbeResult | null = null,
  error: string | null = null,
): boolean {
  return (
    !error &&
    (!probe ||
      (probe.profile_id === profile.profile_id &&
        probe.profile_revision === profile.revision &&
        probe.capabilities.endpoint_reachable &&
        probe.capabilities.model_available &&
        probe.capabilities.embedding_available)) &&
    profile.credential_state !== "unavailable" &&
    !(profile.has_configured_api_key && !profile.has_api_key) &&
    profile.capabilities.endpoint_reachable &&
    profile.capabilities.model_available &&
    profile.capabilities.embedding_available
  );
}

function endpointLabel(value: string | null): string {
  if (!value) return "No endpoint provided";
  try {
    const endpoint = new URL(value);
    if (
      endpoint.username ||
      endpoint.password ||
      endpoint.search ||
      endpoint.hash
    )
      return "Endpoint pending verification";
    return `${endpoint.origin}${endpoint.pathname.replace(/\/$/, "")}`;
  } catch {
    return "Endpoint pending verification";
  }
}

export interface ModelSettingsActions {
  edit(profileId: string): boolean;
  editProfile(profile: ModelProfile): boolean;
  add(): boolean;
  test(profileId: string): Promise<unknown>;
}

export function ModelSettings({
  controlsRef,
  onProfilesChanged,
  compactSingle = false,
  onSetupComplete,
}: {
  controlsRef?: Ref<ModelSettingsActions>;
  onProfilesChanged?(): void;
  compactSingle?: boolean;
  onSetupComplete?(): void;
} = {}) {
  const models = useModelProfiles();
  const [provider, setProvider] = useState<ProviderKind>("openai_compatible");
  const [providerPreset, setProviderPreset] = useState("custom");
  const [advancedEmbedding, setAdvancedEmbedding] = useState(false);
  const [embeddingMode, setEmbeddingMode] = useState<"local" | "custom">(
    "local",
  );
  const [embeddingModeChanged, setEmbeddingModeChanged] = useState(false);
  const [embeddingProvider, setEmbeddingProvider] = useState<
    "openai_compatible" | "ollama"
  >("openai_compatible");
  const [name, setName] = useState("");
  const [nameEdited, setNameEdited] = useState(false);
  const [baseUrl, setBaseUrl] = useState(PROVIDERS[0].baseUrl);
  const [embeddingBaseUrl, setEmbeddingBaseUrl] = useState("");
  const [model, setModel] = useState("");
  const [embeddingModel, setEmbeddingModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [disableThinking, setDisableThinking] = useState(true);
  const [customOpen, setCustomOpen] = useState(false);
  const [saved, setSaved] = useState<ModelProfile | null>(null);
  const [editing, setEditing] = useState<ModelProfile | null>(null);
  const [saving, setSaving] = useState(false);
  const [fields, setFields] = useState<Partial<Record<FieldName, string>>>({});
  const [saveError, setSaveError] = useState<string | null>(null);
  const [unknownCreation, setUnknownCreation] = useState(false);
  const [setupPhase, setSetupPhase] = useState<"testing" | "activating" | null>(
    null,
  );
  const [setupError, setSetupError] = useState<string | null>(null);
  const flight = useRef(false);
  const alive = useRef(true);
  const savedCard = useRef<HTMLElement>(null);
  const formEl = useRef<HTMLFormElement>(null);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  useLayoutEffect(() => {
    if (saved) savedCard.current?.focus({ preventScroll: true });
  }, [saved]);
  useLayoutEffect(() => {
    const first = Object.keys(fields).find((key) => fields[key as FieldName]);
    if (advancedEmbedding && first?.startsWith("embedding-")) {
      formEl.current
        ?.querySelector<HTMLInputElement>(`[name="${first}"]`)
        ?.focus();
    }
  }, [advancedEmbedding, fields]);
  const selected =
    PROVIDERS.find((item) => item.kind === provider) ?? PROVIDERS[0];
  const presetProfiles = models.profiles.filter(
    (profile) => profile.name === PRESET_PROFILE_NAME,
  );
  // With no preset there is nothing else to configure with, so the form is
  // the page rather than an alternative to it.
  const formOpen = !saved && (!presetProfiles.length || customOpen);
  const savedProfile =
    saved &&
    (models.profiles.find(
      (item) =>
        item.profile_id === saved.profile_id && item.revision >= saved.revision,
    ) ??
      saved);
  const editorLocked =
    saving ||
    models.busy !== null ||
    !!models.pendingCreation ||
    !!models.pendingProbe ||
    models.probeStorageFailed === true ||
    models.creationStorageFailed === true;
  const locked = editorLocked;
  const ownerLocked = editorLocked;
  const formLocked = ownerLocked;
  const editedCurrent =
    editing &&
    models.profiles.find((item) => item.profile_id === editing.profile_id);
  const editChanged =
    !!editing &&
    (!editedCurrent || editedCurrent.revision !== editing.revision);
  const endpointMoved =
    !!editing &&
    (provider !== editing.provider_kind ||
      endpointKey(baseUrl) !== endpointKey(editing.base_url ?? ""));
  const keyRequired =
    selected.keyRequired &&
    !(
      editing &&
      (editing.has_configured_api_key ?? editing.has_api_key) &&
      !endpointMoved
    );
  const suggestedName = (nextModel: string, nextProvider = provider) =>
    nextModel.trim()
      ? `${providerLabel(nextProvider)} · ${nextModel.trim()}`.slice(0, 128)
      : "";
  const isReady = (profile: ModelProfile) =>
    ready(profile, models.probeFor(profile), models.probeErrorFor(profile));

  function editProfile(profile: ModelProfile) {
    if (ownerLocked) return;
    setEditing({ ...profile });
    setEmbeddingModeChanged(false);
    setSaved(null);
    setCustomOpen(true);
    setName(profile.name);
    setNameEdited(true);
    setProvider(profile.provider_kind);
    setProviderPreset(presetForProfile(profile));
    setBaseUrl(profile.base_url ?? "");
    setModel(profile.model);
    setDisableThinking(profile.disable_thinking);
    setApiKey("");
    setAdvancedEmbedding(
      profile.embedding_provider_kind !== BUILTIN_EMBEDDING_KIND &&
        !!profile.embedding_model,
    );
    setEmbeddingMode(
      profile.embedding_provider_kind !== BUILTIN_EMBEDDING_KIND &&
        profile.embedding_model
        ? "custom"
        : "local",
    );
    setEmbeddingProvider(
      profile.embedding_provider_kind === "ollama"
        ? "ollama"
        : "openai_compatible",
    );
    setEmbeddingModel(profile.embedding_model ?? "");
    setEmbeddingBaseUrl(profile.embedding_base_url ?? profile.base_url ?? "");
    setFields({});
    setSaveError(null);
    setSetupError(null);
  }

  function newProfile() {
    if (ownerLocked) return;
    setEditing(null);
    setEmbeddingModeChanged(false);
    setSaved(null);
    setCustomOpen(true);
    setName("");
    setNameEdited(false);
    setProvider("openai_compatible");
    setProviderPreset("custom");
    setBaseUrl(PROVIDERS[0].baseUrl);
    setModel("");
    setApiKey("");
    setDisableThinking(true);
    setAdvancedEmbedding(false);
    setEmbeddingMode("local");
    setEmbeddingModel("");
    setEmbeddingBaseUrl("");
    setFields({});
    setSaveError(null);
    setSetupError(null);
  }

  useImperativeHandle(controlsRef, () => ({
    edit(profileId) {
      const profile = models.profiles.find(
        (item) => item.profile_id === profileId,
      );
      if (!profile || ownerLocked) return false;
      editProfile(profile);
      return true;
    },
    editProfile(profile) {
      if (ownerLocked) return false;
      editProfile(profile);
      return true;
    },
    add() {
      if (ownerLocked) return false;
      newProfile();
      return true;
    },
    async test(profileId) {
      if (
        locked ||
        !models.profiles.some((item) => item.profile_id === profileId)
      )
        throw new Error("model_profile_not_available");
      return models.probeProfile(profileId);
    },
  }));
  const profileSnapshot = JSON.stringify(models.profiles);
  const previousProfileSnapshot = useRef(profileSnapshot);
  useEffect(() => {
    if (previousProfileSnapshot.current === profileSnapshot) return;
    previousProfileSnapshot.current = profileSnapshot;
    onProfilesChanged?.();
  }, [profileSnapshot, onProfilesChanged]);

  async function finishSetup(profile: ModelProfile): Promise<boolean> {
    setSetupPhase("testing");
    const result = await models.probeProfile(profile.profile_id);
    if (!alive.current) return false;
    if (
      result &&
      result.profile_id === profile.profile_id &&
      result.profile_revision !== profile.revision
    ) {
      // A newer revision used to be accepted here, and then activation turned
      // on whatever the configuration had become. The user is told the
      // settings they filled in passed, so this hands them a model they never
      // tested. `ready` below cannot catch it either: by then the probe and
      // the activated profile are both on the new revision and agree.
      setSetupError(
        result.profile_revision > profile.revision
          ? "This configuration was edited during the test; the test ran against the edited version. It is saved but not activated. Test and activate it again."
          : "The test result came from an earlier revision of this configuration. It is saved but not activated. Test and activate it again.",
      );
      return false;
    }
    if (
      !result ||
      result.profile_id !== profile.profile_id ||
      result.profile_revision !== profile.revision ||
      !result.capabilities.endpoint_reachable ||
      !result.capabilities.model_available ||
      !result.capabilities.embedding_available
    ) {
      setSetupError(
        "Configuration saved, but the connection test failed. Review the configuration and retry; no conversation history has been sent.",
      );
      setFields(probeFieldErrors(result));
      return false;
    }
    setSetupPhase("activating");
    const active = await models.activate(profile.profile_id);
    if (!alive.current) return false;
    if (
      !active ||
      active.profile_id !== profile.profile_id ||
      !active.is_active ||
      !ready(active, result)
    ) {
      setSetupError(
        "Configuration saved, but activation was not confirmed. Review the configuration and retry.",
      );
      return false;
    }
    if (profile.name !== PRESET_PROFILE_NAME) setSaved(active);
    onSetupComplete?.();
    return true;
  }

  async function resumeSetup(profile: ModelProfile, key?: string) {
    if (flight.current || locked) return;
    flight.current = true;
    setSaving(true);
    setSetupError(null);
    try {
      const updated =
        key === undefined
          ? profile
          : await models.update(profile.profile_id, {
              expected_revision: profile.revision,
              api_key: key,
            });
      if (!alive.current || !updated) return;
      await finishSetup(updated);
      return updated;
    } catch {
      if (alive.current)
        setSetupError(
          "The configuration has not finished testing and activation. Review it and retry.",
        );
    } finally {
      flight.current = false;
      if (alive.current) {
        setSaving(false);
        setSetupPhase(null);
      }
    }
  }

  /**
   * Saving the key without testing it. The preset card's own submit, kept
   * apart from `resumeSetup` because a save that reaches out to the endpoint
   * is a different act from a save that does not, and only the user gets to
   * decide which one they are performing.
   */
  async function saveKey(profile: ModelProfile, key: string) {
    if (flight.current || locked) return;
    flight.current = true;
    setSaving(true);
    setSetupError(null);
    try {
      return await models.update(profile.profile_id, {
        expected_revision: profile.revision,
        api_key: key,
      });
    } finally {
      flight.current = false;
      if (alive.current) setSaving(false);
    }
  }

  /**
   * `thenSetup` is the whole difference between the form's two buttons.
   *
   * Plain save is the default, including the Enter key, because testing sends
   * a request to a third-party endpoint and activation cancels queued work
   * bound to the profile being replaced (`memory_processing` drops tasks whose
   * snapshot no longer matches the active profile). Neither belongs to a
   * keystroke the user reads as "write this down".
   */
  async function persist(thenSetup: boolean) {
    if (
      flight.current ||
      locked ||
      editChanged ||
      (!editing && unknownCreation)
    )
      return;
    const errors: Partial<Record<FieldName, string>> = {};
    if (!model.trim()) errors.model = "Enter the generation model name.";
    if (!validEndpoint(baseUrl))
      errors["base-url"] = "Enter a complete http:// or https:// endpoint.";
    const credentialError = keyError(apiKey, keyRequired);
    if (credentialError) errors["api-key"] = credentialError;
    if (embeddingMode === "custom") {
      if (!embeddingModel.trim())
        errors["embedding-model"] = "Enter the embedding model name.";
      if (!validEndpoint(embeddingBaseUrl))
        errors["embedding-base-url"] =
          "Enter a complete endpoint for the embedding model.";
    }
    setFields(errors);
    setSaveError(null);
    setSetupError(null);
    const first = Object.keys(errors)[0];
    if (first) {
      if (first.startsWith("embedding-") && !advancedEmbedding) {
        setAdvancedEmbedding(true);
        return;
      }
      formEl.current
        ?.querySelector<HTMLInputElement>(`[name="${first}"]`)
        ?.focus();
      return;
    }
    flight.current = true;
    setSaving(true);
    const input: ModelProfileInput = {
      name: name.trim() || suggestedName(model),
      provider_kind: provider,
      model,
      disable_thinking: disableThinking,
      ...(embeddingMode === "custom"
        ? {
            embedding_model: embeddingModel,
            embedding_provider_kind: embeddingProvider,
            embedding_base_url: embeddingBaseUrl,
          }
        : {}),
      ...(provider === "hy_default" ? {} : { base_url: baseUrl }),
      ...(apiKey ? { api_key: apiKey } : {}),
    };
    try {
      const patch: ModelProfilePatch = editing
        ? { expected_revision: editing.revision }
        : { expected_revision: 0 };
      if (editing) {
        if (input.name !== editing.name) patch.name = input.name;
        if (input.model !== editing.model) patch.model = input.model;
        if (input.base_url !== editing.base_url)
          patch.base_url = input.base_url;
        if (input.disable_thinking !== editing.disable_thinking)
          patch.disable_thinking = input.disable_thinking;
        if (apiKey) patch.api_key = apiKey;
        if (embeddingMode === "custom") {
          if (embeddingModel !== editing.embedding_model)
            patch.embedding_model = embeddingModel;
          if (embeddingProvider !== editing.embedding_provider_kind)
            patch.embedding_provider_kind = embeddingProvider;
          if (
            embeddingBaseUrl !==
            (editing.embedding_base_url ?? editing.base_url ?? "")
          )
            patch.embedding_base_url = embeddingBaseUrl;
        } else if (
          embeddingModeChanged &&
          editing.embedding_provider_kind !== BUILTIN_EMBEDDING_KIND
        ) {
          // Core clears the remote endpoint when local_onnx is selected.
          patch.embedding_provider_kind = BUILTIN_EMBEDDING_KIND;
          patch.embedding_model = "bge-small-zh-v1.5";
          patch.embedding_dimensions = 512;
        }
      }
      const created =
        editing && provider === editing.provider_kind
          ? await models.update(editing.profile_id, patch)
          : await models.create(input);
      if (!alive.current) return;
      if (created) {
        if (thenSetup) {
          // The probe decides whether this form closes. A rejected key or a
          // failed endpoint leaves the filled-in form on screen with the
          // reason; only a passing test returns to the saved card.
          const passed = await finishSetup(created);
          if (!alive.current) return;
          if (passed) {
            setSaved(created);
            setEditing(null);
            setCustomOpen(false);
            setApiKey("");
          }
        } else {
          setSaved(created);
          setEditing(null);
          setCustomOpen(false);
          setApiKey("");
        }
      } else {
        setSaveError(
          created === null
            ? "Configuration not saved; check the model address and parameters, then retry. Your entries are kept."
            : "The configuration was not saved. Your entries are kept; review and retry.",
        );
      }
    } catch {
      if (alive.current) {
        setSaveError(
          "The configuration was not saved. Your entries are kept; review and retry.",
        );
      }
    } finally {
      flight.current = false;
      if (alive.current) {
        setSaving(false);
        setSetupPhase(null);
      }
    }
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void persist(false);
  }

  return (
    <section className="model-settings" aria-labelledby="models-title">
      <div className="section-heading">
        <div>
          <p className="section-index">02</p>
          <h2 id="models-title">Memory models</h2>
        </div>
        <p>
          You only need to configure the generation model. Embedding is built in
          on this machine; System 1 and System 2 do not get separate model
          pickers.
        </p>
      </div>

      {models.error ? (
        <p className="error-banner" role="alert">
          {models.error}
        </p>
      ) : null}
      {(models.pendingProbe || models.probeStorageFailed) && (
        <div role="status" className="model-refresh-notice">
          <p>
            {models.probeStorageFailed
              ? "The test recovery record cannot be read; no new test is sent."
              : "The last model test result is pending verification; do not change or activate the configuration."}
          </p>
          {models.pendingProbe && (
            <button
              type="button"
              disabled={
                models.busy !== null || saving || models.probeStorageFailed
              }
              onClick={async () => {
                const pending = models.pendingProbe;
                if (!pending) return;
                await models.probeProfile(pending.profileId);
                if (alive.current) onProfilesChanged?.();
              }}
            >
              Verify test result
            </button>
          )}
        </div>
      )}
      {(models.pendingCreation || models.creationStorageFailed) && (
        <div role="status" className="model-refresh-notice">
          <p>
            {models.creationMissing
              ? "The previous creation record was not found. After you confirm dropping the old record you can reconfigure; the system will not retry automatically."
              : "The last model creation result is pending verification; no new operation starts."}
          </p>
          <button
            type="button"
            disabled={models.busy !== null || saving}
            onClick={async () => {
              const recovered = await models.reconcileCreation();
              if (!alive.current || recovered === undefined) return;
              setUnknownCreation(false);
              setSaveError(null);
              setEditing(null);
              setApiKey("");
              setSaved(recovered);
              onProfilesChanged?.();
            }}
          >
            Verify creation result
          </button>
          {models.pendingCreation && (
            <button
              type="button"
              className="secondary"
              disabled={models.busy !== null || saving}
              onClick={async () => {
                if (!(await models.discardMissingCreation()) || !alive.current)
                  return;
                setUnknownCreation(false);
                setSaveError(null);
              }}
            >
              Drop the old record and reconfigure
            </button>
          )}
        </div>
      )}
      {setupPhase && (
        <p className="model-test-notice" role="status">
          {setupPhase === "testing"
            ? "Configuration saved; testing the connection..."
            : "Connection passed; activating the model..."}
        </p>
      )}
      {setupError && (
        <p className="model-save-error" role="alert">
          {setupError}
        </p>
      )}
      {models.refreshError && (
        <div className="model-refresh-notice" role="status">
          <p>{models.refreshError}</p>
          <button
            type="button"
            className="secondary"
            disabled={locked}
            onClick={() => void models.refreshProfiles()}
          >
            <RefreshCw size={16} aria-hidden="true" />
            Refresh the configuration list
          </button>
        </div>
      )}

      {/* The paste-one-key path goes first when it exists. Behind it sits the
       * full form, which asks for a protocol, a base URL, a timeout and a
       * concurrency limit — none of which the person with a preset needs to
       * have an opinion about. */}
      {!compactSingle && presetProfiles.length ? (
        <div className="profile-list preset-first" aria-live="polite">
          {presetProfiles.map((profile) => (
            <PresetProfileCard
              key={profile.profile_id}
              profile={profile}
              busy={locked}
              onSaveKey={(key) => saveKey(profile, key)}
              onSaveKeyAndSetup={(key) => resumeSetup(profile, key)}
              onContinue={() => resumeSetup(profile)}
              onTest={() => models.probeProfile(profile.profile_id)}
              onActivate={() => models.activate(profile.profile_id)}
              onRemove={() => models.remove(profile.profile_id)}
              probe={models.probeFor(profile)}
              probeError={models.probeErrorFor(profile)}
            />
          ))}
          <button
            type="button"
            className="quiet"
            aria-expanded={customOpen}
            disabled={locked}
            onClick={() => {
              if (customOpen) {
                setEditing(null);
                setCustomOpen(false);
                setApiKey("");
              } else newProfile();
            }}
          >
            Add a custom model
          </button>
        </div>
      ) : null}

      {!compactSingle && savedProfile && (
        <article
          className="model-saved-card"
          aria-label="Saved model configurations"
          tabIndex={-1}
          ref={savedCard}
          data-tour-id="model-form"
        >
          <div className="model-saved-heading">
            <Check size={18} aria-hidden="true" />
            <strong>
              {isReady(savedProfile) && savedProfile.is_active
                ? "Saved and activated"
                : isReady(savedProfile)
                  ? "Saved, connection passed"
                  : models.probeFor(savedProfile)
                    ? "Saved, test failed"
                    : "Saved, pending test"}
            </strong>
          </div>
          <h3>{savedProfile.name}</h3>
          <p>
            {savedProfile.model} - {endpointLabel(savedProfile.base_url)} -
            Revision {savedProfile.revision}
          </p>
          {savedProfile.credential_state === "unavailable" && (
            <p role="status">
              MemHub needs access to the keychain. Choose "Always Allow" when
              the system asks; saved configurations are not lost.
            </p>
          )}
          <p>
            The test sends only test content to this endpoint, never
            conversation history.
          </p>
          <div className="source-actions">
            <button
              type="button"
              disabled={locked}
              onClick={() => void resumeSetup(savedProfile)}
            >
              {setupError ? "Retry test and activation" : "Test and activate"}
            </button>
            <button
              type="button"
              className="secondary"
              disabled={locked}
              onClick={() => editProfile(savedProfile)}
            >
              <Pencil size={16} aria-hidden="true" />
              Edit configuration
            </button>
            <button
              type="button"
              className="quiet"
              disabled={locked}
              onClick={newProfile}
            >
              <Plus size={16} aria-hidden="true" />
              Add configuration
            </button>
          </div>
          <ProbeFeedback
            profile={savedProfile}
            result={models.probeFor(savedProfile)}
            error={models.probeErrorFor(savedProfile)}
          />
        </article>
      )}

      {formOpen ? (
        <form
          className="model-form"
          data-tour-id="model-form"
          ref={formEl}
          onSubmit={submit}
          noValidate
          aria-busy={saving}
        >
          {editing && (
            <p className="model-test-notice">
              Edit {editing.name} - Revision {editing.revision}. After changes,
              test again and set it as the current model; content authorization
              does not update automatically.
            </p>
          )}
          {editChanged && (
            <p role="alert" className="model-save-error">
              The configuration changed elsewhere; your draft is kept. Cancel
              the edit and reopen it to verify.
            </p>
          )}
          {saveError && (
            <p className="model-save-error" role="alert">
              {saveError}
            </p>
          )}
          {unknownCreation && !editing && (
            <button
              type="button"
              disabled={formLocked}
              onClick={() => void models.refreshProfiles()}
            >
              Refresh the configuration list
            </button>
          )}
          <div className="model-form-field">
            Provider preset
            <OptionMenu
              ariaLabel="Provider preset"
              describedBy="model-preset-help"
              value={providerPreset}
              disabled={formLocked}
              options={[
                ...PROVIDER_PRESETS.map((item) => ({
                  value: item.id,
                  label: item.label,
                })),
                { value: "custom", label: "Custom compatible service" },
              ]}
              onChange={(value) => {
                const next = PROVIDER_PRESETS.find((item) => item.id === value);
                setProviderPreset(value);
                if (!next) return;
                setProvider(next.provider);
                setBaseUrl(next.baseUrl);
                setModel(next.model);
                setApiKey("");
                setDisableThinking(true);
                if (editing) {
                  setName(next.label);
                  setNameEdited(true);
                } else if (!nameEdited) {
                  setName(suggestedName(next.model, next.provider));
                }
                setFields({});
                setSaveError(null);
              }}
            />
            <small id="model-preset-help">
              This only fills in the endpoint and a model name for you; it does
              not mean the model is available right now. Check it against the
              provider's model list before saving.
            </small>
          </div>
          <label>
            Protocol
            <select
              name="provider"
              aria-label="Protocol"
              aria-describedby="model-provider-help"
              value={provider}
              disabled={formLocked}
              onChange={(event) => {
                const next = event.target.value as ProviderKind;
                setProvider(next);
                setProviderPreset("custom");
                if (!nameEdited) setName(suggestedName(model, next));
              }}
            >
              {/* Only what the backend will actually activate. These used to be
               * listed and disabled, which put two dead ends on screen and made
               * the reader parse a parenthesis to find out they were dead. */}
              {PROVIDERS.filter((item) => item.activatable).map((item) => (
                <option key={item.kind} value={item.kind}>
                  {item.label}
                </option>
              ))}
            </select>
            <small id="model-provider-help">
              Keep this aligned with the protocol offered by the model service.
            </small>
          </label>
          <label>
            Configuration name
            <input
              name="name"
              aria-label="Configuration name"
              aria-describedby="model-name-help"
              autoComplete="off"
              maxLength={128}
              value={name}
              disabled={formLocked}
              onChange={(event) => {
                setNameEdited(true);
                setName(event.target.value);
              }}
            />
            <small id="model-name-help">
              Used to tell configurations apart; it does not affect model
              requests.
            </small>
          </label>
          <label>
            Generation model name
            <input
              name="model"
              aria-label="Generation model name"
              aria-describedby={`model-identifier-help${fields.model ? " model-field-error" : ""}`}
              aria-invalid={!!fields.model}
              autoComplete="off"
              required
              maxLength={256}
              value={model}
              disabled={formLocked}
              onChange={(event) => {
                setModel(event.target.value);
                setProviderPreset("custom");
                if (!nameEdited) setName(suggestedName(event.target.value));
              }}
              placeholder="e.g. gpt-4o-mini..."
            />
            <small id="model-identifier-help">
              Enter the full model identifier provided by the provider.
            </small>
            {fields.model && (
              <small id="model-field-error" className="model-field-error">
                {fields.model}
              </small>
            )}
          </label>
          {/* Disclosure only changes visibility; mode determines what is saved. */}
          <div className="model-advanced">
            <button
              type="button"
              className="quiet"
              aria-expanded={advancedEmbedding}
              disabled={formLocked}
              onClick={() => setAdvancedEmbedding(!advancedEmbedding)}
            >
              Advanced: configure the embedding model yourself
            </button>
            {advancedEmbedding ? (
              <>
                <label>
                  Embedding configuration mode
                  <select
                    name="embedding-mode"
                    value={embeddingMode}
                    disabled={formLocked}
                    onChange={(event) => {
                      setEmbeddingMode(
                        event.target.value as "local" | "custom",
                      );
                      setEmbeddingModeChanged(true);
                      setFields((previous) => ({
                        ...previous,
                        "embedding-model": undefined,
                        "embedding-base-url": undefined,
                      }));
                    }}
                  >
                    <option value="local">Built-in local</option>
                    <option value="custom">Custom embedding model</option>
                  </select>
                </label>
                {embeddingMode === "custom" ? (
                  <>
                    <label>
                      Embedding model name
                      <input
                        name="embedding-model"
                        autoComplete="off"
                        required
                        maxLength={256}
                        value={embeddingModel}
                        disabled={formLocked}
                        aria-invalid={!!fields["embedding-model"]}
                        aria-describedby={
                          fields["embedding-model"]
                            ? "model-embedding-error"
                            : undefined
                        }
                        onChange={(event) =>
                          setEmbeddingModel(event.target.value)
                        }
                        placeholder="e.g. text-embedding-3-small..."
                      />
                      {fields["embedding-model"] && (
                        <small
                          id="model-embedding-error"
                          className="model-field-error"
                        >
                          {fields["embedding-model"]}
                        </small>
                      )}
                    </label>
                    <label>
                      Embedding model source
                      <select
                        name="embedding-provider"
                        value={embeddingProvider}
                        disabled={formLocked}
                        onChange={(event) => {
                          const next = event.target.value as
                            | "openai_compatible"
                            | "ollama";
                          setEmbeddingProvider(next);
                        }}
                      >
                        <option value="ollama">Local Ollama</option>
                        <option value="openai_compatible">
                          Same-origin OpenAI-compatible (shares the API Key)
                        </option>
                      </select>
                    </label>
                    <label>
                      Embedding model Base URL
                      <input
                        name="embedding-base-url"
                        type="url"
                        inputMode="url"
                        autoComplete="off"
                        required
                        value={embeddingBaseUrl}
                        disabled={formLocked}
                        placeholder={
                          embeddingProvider === "ollama"
                            ? "http://127.0.0.1:11434"
                            : baseUrl
                        }
                        aria-invalid={!!fields["embedding-base-url"]}
                        aria-describedby={
                          fields["embedding-base-url"]
                            ? "model-embedding-url-error"
                            : undefined
                        }
                        onChange={(event) =>
                          setEmbeddingBaseUrl(event.target.value)
                        }
                      />
                      {fields["embedding-base-url"] && (
                        <small
                          id="model-embedding-url-error"
                          className="model-field-error"
                        >
                          {fields["embedding-base-url"]}
                        </small>
                      )}
                    </label>
                    {embeddingProvider === "openai_compatible" ? (
                      <small>
                        To avoid leaking credentials, the embedding address must
                        share the generation model's origin.
                      </small>
                    ) : null}
                  </>
                ) : (
                  <small>
                    Uses the built-in local embedding model; nothing is sent to
                    a remote embedding service.
                  </small>
                )}
              </>
            ) : null}
            {!advancedEmbedding && embeddingMode === "custom" ? (
              <p className="model-help">
                Custom embedding: {embeddingModel || "No model entered yet"} -{" "}
                {embeddingBaseUrl || "No address entered yet"}
              </p>
            ) : null}
          </div>
          <label>
            Base URL
            <input
              name="base-url"
              aria-label="Base URL"
              aria-describedby={`model-url-help${fields["base-url"] ? " model-url-error" : ""}`}
              aria-invalid={!!fields["base-url"]}
              type="url"
              inputMode="url"
              autoComplete="off"
              required={provider !== "hy_default"}
              disabled={formLocked || provider === "hy_default"}
              value={baseUrl}
              onChange={(event) => {
                setBaseUrl(event.target.value);
                setProviderPreset("custom");
              }}
            />
            <small id="model-url-help">
              Enter the service endpoint, not the chat webpage address.
            </small>
            {fields["base-url"] && (
              <small id="model-url-error" className="model-field-error">
                {fields["base-url"]}
              </small>
            )}
          </label>
          <label>
            API Key{keyRequired ? "" : " (optional)"}
            <input
              name="api-key"
              aria-label={`API Key${keyRequired ? "" : " (optional)"}`}
              aria-describedby={`model-key-help${fields["api-key"] ? " model-key-error" : ""}`}
              aria-invalid={!!fields["api-key"]}
              type="password"
              autoComplete="off"
              required={keyRequired}
              value={apiKey}
              disabled={formLocked}
              onChange={(event) => setApiKey(event.target.value)}
              placeholder="Not shown again after saving..."
            />
            <small id="model-key-help">
              {editing && !endpointMoved
                ? "Leave blank to keep the saved key; a new key requires re-testing and activation."
                : editing && keyRequired
                  ? "After switching providers, enter a new API Key; the saved key is not carried to the new address."
                  : "Stored in the system secure store; never shipped with the installer."}
            </small>
            {fields["api-key"] && (
              <small id="model-key-error" className="model-field-error">
                {fields["api-key"]}
              </small>
            )}
          </label>
          <label>
            <input
              name="disable-thinking"
              type="checkbox"
              checked={disableThinking}
              disabled={formLocked}
              onChange={(event) => setDisableThinking(event.target.checked)}
            />
            Disable the chain of thought (sends the thinking-off parameter to
            supported vendors)
          </label>
          <div className="form-submit">
            {!compactSingle && (
              <button
                type="submit"
                disabled={
                  locked || editChanged || (!editing && unknownCreation)
                }
              >
                {editing ? "Save changes" : "Save configuration"}
              </button>
            )}
            <button
              type="button"
              disabled={locked || editChanged || (!editing && unknownCreation)}
              onClick={() => void persist(true)}
            >
              {setupPhase === "testing"
                ? "Testing the connection..."
                : setupPhase === "activating"
                  ? "Activating the configuration..."
                  : compactSingle
                    ? "Test and finish setup"
                    : "Save, test and activate"}
            </button>
            {editing && (
              <button
                type="button"
                className="secondary"
                disabled={formLocked}
                onClick={() => {
                  setSaved(editedCurrent ?? editing);
                  setEditing(null);
                  setApiKey("");
                  setSaveError(null);
                  setFields({});
                }}
              >
                Cancel edit
              </button>
            )}
            <small>
              Keys are stored in the system secure store; tests only send test
              content.
            </small>
          </div>
        </form>
      ) : null}

      {!compactSingle && (
        <div className="profile-list" aria-live="polite">
          {models.profiles
            .filter(
              (profile) =>
                profile.name !== PRESET_PROFILE_NAME &&
                profile.profile_id !== savedProfile?.profile_id,
            )
            .map((profile) => (
              <article key={profile.profile_id} className="profile-row">
                {profile.credential_state === "unavailable" && (
                  <p className="error-banner" role="status">
                    MemHub needs access to the keychain. Choose "Always Allow"
                    when the system asks; saved configurations are not lost.
                  </p>
                )}
                <div>
                  <strong>{profile.name}</strong>
                  <p>
                    {providerLabel(profile.provider_kind)} · {profile.model} ·
                    {/* Core fills in the built-in vector model on every profile,
                     * so printing it unconditionally put a line about
                     * bge-small-zh-v1.5 under a profile whose owner never chose
                     * it. Only a vector model the user went out of their way to
                     * set is worth the room. */}
                    {profile.embedding_model &&
                    profile.embedding_provider_kind &&
                    profile.embedding_provider_kind !== BUILTIN_EMBEDDING_KIND
                      ? ` Embedding: ${profile.embedding_model} (${providerLabel(
                          profile.embedding_provider_kind,
                        )}) ·`
                      : ""}{" "}
                    revision {profile.revision}
                  </p>
                </div>
                <span>
                  {profile.is_active ? "Currently in use" : "Not activated"}
                </span>
                <p className="model-test-notice">
                  Testing target: {endpointLabel(profile.base_url)}. Only test
                  content is sent; conversation history is not.
                </p>
                <div className="source-actions">
                  <button
                    type="button"
                    disabled={locked}
                    onClick={() => void resumeSetup(profile)}
                  >
                    {setupError
                      ? "Retry test and activation"
                      : "Test and activate"}
                  </button>
                  <button
                    type="button"
                    className="secondary"
                    disabled={locked}
                    onClick={() => editProfile(profile)}
                  >
                    <Pencil size={16} aria-hidden="true" />
                    Edit configuration
                  </button>
                  <button
                    type="button"
                    className="secondary"
                    disabled={locked}
                    onClick={() => models.probeProfile(profile.profile_id)}
                  >
                    Test connection
                  </button>
                  {!profile.is_active ? (
                    <>
                      <button
                        type="button"
                        disabled={locked || !isReady(profile)}
                        onClick={() => models.activate(profile.profile_id)}
                      >
                        Set as current model
                      </button>
                      <button
                        type="button"
                        className="quiet"
                        disabled={locked}
                        onClick={() => models.remove(profile.profile_id)}
                      >
                        Delete
                      </button>
                    </>
                  ) : null}
                </div>
                <ProbeFeedback
                  profile={profile}
                  result={models.probeFor(profile)}
                  error={models.probeErrorFor(profile)}
                />
              </article>
            ))}
        </div>
      )}
    </section>
  );
}

/**
 * The profile the build already filled in, showing only the field it could not.
 *
 * Base URL and model name are deliberately not on the surface. A user who can
 * see them assumes they are things to decide, and the whole point of a preset
 * is that they are not.
 */
function PresetProfileCard({
  profile,
  busy,
  canEnable = true,
  discloseRecipients = false,
  onSaveKey,
  onSaveKeyAndSetup,
  onContinue,
  onTest,
  onActivate,
  onRemove,
  probe,
  probeError,
}: {
  profile: ModelProfile;
  busy: boolean;
  canEnable?: boolean;
  discloseRecipients?: boolean;
  onSaveKey: (key: string) => Promise<unknown>;
  onSaveKeyAndSetup: (key: string) => Promise<unknown>;
  onContinue: () => Promise<unknown>;
  onTest: () => Promise<unknown>;
  onActivate: () => Promise<unknown>;
  onRemove: () => void;
  probe: ModelProbeResult | null;
  probeError: string | null;
}) {
  const [key, setKey] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const flight = useRef(false);
  const alive = useRef(true);
  const keyForm = useRef<HTMLFormElement>(null);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  const keyLabel = presetKeyLabel();
  const applyHref = applyUrl(profile.base_url);

  /**
   * Submitting stores the key and stops there; the second button is the one
   * that goes on to contact the endpoint. Same split as the full form, for the
   * same reason — pasting a secret and spending it are separate decisions.
   */
  async function save(thenSetup: boolean) {
    if (busy || flight.current) return;
    const invalid = keyError(key, true);
    setError(invalid ?? null);
    if (invalid) {
      keyForm.current?.querySelector<HTMLInputElement>("input")?.focus();
      return;
    }
    flight.current = true;
    setSaving(true);
    try {
      const saved = await (thenSetup ? onSaveKeyAndSetup(key) : onSaveKey(key));
      if (alive.current) {
        if (saved) setKey("");
        else
          setError("The key was not saved. Your entry is kept; you can retry.");
      }
    } catch {
      if (alive.current)
        setError("The key was not saved. Your entry is kept; you can retry.");
    } finally {
      flight.current = false;
      if (alive.current) setSaving(false);
    }
  }

  return (
    <section className="preset-profile" aria-label={profile.name}>
      <div>
        <strong>{profile.name}</strong>
        <p>{presetStatus(profile)}</p>
      </div>
      {discloseRecipients && (
        <p>
          Generation recipient: {endpointLabel(profile.base_url)}; embedding
          recipient:
          {profile.embedding_provider_kind === BUILTIN_EMBEDDING_KIND
            ? "Local"
            : endpointLabel(profile.embedding_base_url ?? profile.base_url)}
          . Once this configuration is activated, history and future allowed
          sources follow the selections above.
        </p>
      )}
      {profile.credential_state === "unavailable" && (
        <p role="status">
          MemHub needs access to the keychain. Choose "Always Allow" when the
          system asks; saved configurations are not lost.
        </p>
      )}
      {profile.has_api_key ? null : (
        <form
          className="preset-key"
          ref={keyForm}
          onSubmit={(event) => {
            event.preventDefault();
            void save(false);
          }}
          noValidate
        >
          <label>
            {keyLabel}
            <input
              name="preset-api-key"
              type="password"
              autoComplete="off"
              required
              disabled={busy || saving}
              aria-invalid={!!error}
              aria-describedby={
                error ? `preset-key-error-${profile.profile_id}` : undefined
              }
              value={key}
              onChange={(event) => setKey(event.target.value)}
              placeholder="Not shown again after saving..."
            />
            {error && (
              <small
                id={`preset-key-error-${profile.profile_id}`}
                className="model-field-error"
                role="alert"
              >
                {error}
              </small>
            )}
          </label>
          <button type="submit" disabled={busy || saving}>
            Save key
          </button>
          <button
            type="button"
            className="secondary"
            disabled={busy || saving || !canEnable}
            onClick={() => void save(true)}
          >
            Save, test and activate
          </button>
          {applyHref ? (
            <a href={applyHref} target="_blank" rel="noreferrer">
              No key yet? Get one
            </a>
          ) : null}
        </form>
      )}
      <ProbeFeedback profile={profile} result={probe} error={probeError} />
      {profile.has_api_key && !profile.is_active ? (
        <div className="source-actions">
          <button
            type="button"
            disabled={busy || !canEnable}
            onClick={onContinue}
          >
            Test and activate
          </button>
          <p className="model-test-notice">
            Testing target: {endpointLabel(profile.base_url)}. Only test content
            is sent; conversation history is not.
          </p>
          <button type="button" disabled={busy} onClick={onTest}>
            Test connection
          </button>
          <button
            type="button"
            disabled={busy || !canEnable || !ready(profile, probe, probeError)}
            onClick={onActivate}
          >
            Set as current model
          </button>
          <button
            type="button"
            className="quiet"
            disabled={busy}
            onClick={onRemove}
          >
            Delete
          </button>
        </div>
      ) : null}
      {profile.has_api_key ? null : (
        // Same container as the other case. On its own it rendered centred
        // and unattached, reading as a stray control rather than this card's
        // action. It is a real one: a build that ships a preset nobody here
        // can use should be possible to clear away, and the seed is claimed
        // once so it does not come back.
        <div className="source-actions">
          <button
            type="button"
            className="quiet"
            disabled={busy}
            onClick={onRemove}
          >
            Delete this card
          </button>
        </div>
      )}
    </section>
  );
}

/**
 * Waiting for a key and configured wrongly read as the same thing to a user
 * unless they are worded apart. Told apart the only way the contract allows:
 * a profile with no key cannot have been probed, so there is nothing to
 * diagnose yet — only something to paste.
 */
function presetStatus(profile: ModelProfile): string {
  if (profile.credential_state === "unavailable")
    return "The key is configured; its current read status needs confirmation.";
  if (!profile.has_api_key) {
    return "Waiting for a key: everything else is filled in; paste the key and it works.";
  }
  if (profile.is_active) {
    return "Currently in use";
  }
  if (!profile.capabilities.model_available) {
    return "Key saved; this model has not been tested yet.";
  }
  return "Not activated";
}

function ProbeFeedback({
  profile,
  result,
  error,
}: {
  profile: ModelProfile;
  result: ModelProbeResult | null;
  error: string | null;
}) {
  const matching =
    result?.profile_id === profile.profile_id &&
    result.profile_revision === profile.revision
      ? result
      : null;
  if (!matching && !error) return null;
  return (
    <section
      className="probe-result"
      aria-label={`${profile.name} revision ${profile.revision} test result`}
    >
      <strong>
        {profile.name} - Revision {profile.revision} - Connection test
      </strong>
      <p>
        Testing target: {endpointLabel(profile.base_url)} - {profile.model}
      </p>
      {error && <p role="alert">{error}</p>}
      {matching && (
        <ul>
          {matching.stages.map((stage) => (
            <li key={stage.stage}>
              {stage.stage}: {probeLabel(stage.code)} · {stage.latency_ms}ms
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function providerLabel(provider: ProviderKind): string {
  if (provider === "local_onnx") {
    return "Built-in bge-small-zh-v1.5";
  }
  return PROVIDERS.find((item) => item.kind === provider)?.label ?? provider;
}

export function probeLabel(code: ProbeCode): string {
  const labels: Record<ProbeCode, string> = {
    ok: "Passed",
    hy_default_unavailable: "HY hosted model not available yet",
    missing_api_key: "Missing API Key",
    unreachable: "Service unreachable",
    unauthorized:
      "Authentication failed; update the key and test again, processing resumes automatically",
    rate_limited: "Rate limited",
    insufficient_balance:
      "The model account balance is exhausted; top up or change the key and test again, processing resumes automatically",
    model_not_found: "Model not found",
    model_list_unsupported:
      "This gateway does not expose a model list; a generation call verifies instead",
    invalid_response: "The service response does not match the protocol",
    structured_output_unsupported: "Structured output failed",
    embedding_model_required: "Embedding model required",
    embedding_unavailable: "Embedding model unavailable",
  };
  return labels[code];
}
