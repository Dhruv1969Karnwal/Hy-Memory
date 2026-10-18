import type {
  ModelProbeResult,
  ModelProfile,
  ModelProfileInput,
} from "@one-memory/contracts";
import { useRef, useState } from "react";
import { getModelProfiles } from "./api";
import {
  type OnboardingModelDraft,
  OnboardingModelStep,
} from "./OnboardingModelStep";
import { PROVIDER_PRESETS } from "./provider-presets";
import { traceUiAction } from "./ui-trace";
import { useModelProfiles } from "./use-model-profiles";

const defaultPreset = PROVIDER_PRESETS.find(
  (preset) => preset.id === "deepseek",
);
const initial: OnboardingModelDraft = {
  provider_id: "deepseek",
  name: "DeepSeek",
  provider_kind: "openai_compatible",
  model: defaultPreset?.model ?? "",
  base_url: defaultPreset?.baseUrl ?? "",
  api_key: "",
  disable_thinking: true,
  embedding_mode: "local",
  embedding_provider_kind: "openai_compatible",
  embedding_model: "",
  embedding_base_url: "",
  timeout_seconds: "90",
  max_concurrency: "4",
  context_window: "",
};

export function probeMatchesProfile(
  original: ModelProfile,
  current: ModelProfile | undefined,
  probe: ModelProbeResult | undefined | null,
): current is ModelProfile {
  return Boolean(
    current &&
      probe &&
      probe.profile_id === original.profile_id &&
      probe.profile_revision === current.revision &&
      probe.capabilities.endpoint_reachable &&
      probe.capabilities.model_available &&
      probe.capabilities.embedding_available &&
      probe.stages.every((stage) => stage.ok),
  );
}

export function OnboardingModelController({
  sourceNames,
  importedConversationCount = 0,
  onBack,
  onSkip,
  onEnabled,
  connectionAlreadyApplied = false,
}: {
  sourceNames: readonly string[];
  importedConversationCount?: number;
  onBack(): void;
  onSkip?(): void;
  onEnabled?(): void;
  connectionAlreadyApplied?: boolean;
}) {
  const models = useModelProfiles();
  const [draft, setDraft] = useState(initial);
  const [advanced, setAdvanced] = useState(false);
  const [saved, setSaved] = useState<ModelProfile | null>(null);
  const [phase, setPhase] = useState<
    "idle" | "saving" | "testing" | "enabling"
  >("idle");
  const [error, setError] = useState<string | null>(null);
  const [fields, setFields] = useState<
    Partial<Record<keyof OnboardingModelDraft, string>>
  >({});
  const [probe, setProbe] = useState<{
    state: "passed" | "failed" | "stale";
    message: string;
  } | null>(null);
  const flight = useRef(false);

  function edit(patch: Partial<OnboardingModelDraft>) {
    if (flight.current) return;
    if (patch.provider_id && patch.provider_id !== draft.provider_id) {
      const preset = PROVIDER_PRESETS.find(
        (item) => item.id === patch.provider_id,
      );
      patch = {
        ...patch,
        provider_kind: preset?.provider ?? "openai_compatible",
        base_url: preset?.baseUrl ?? "",
        model: preset?.model ?? "",
        name: preset?.label ?? "",
        api_key: "",
      };
      setSaved(null);
      setAdvanced(!preset?.model);
    }
    setDraft((value) => ({ ...value, ...patch }));
    setFields({});
    if (probe)
      setProbe({
        state: "stale",
        message: "The configuration changed; it needs a new test.",
      });
  }

  function input(): ModelProfileInput | null {
    const next: Partial<Record<keyof OnboardingModelDraft, string>> = {};
    const integer = (
      key: "timeout_seconds" | "max_concurrency" | "context_window",
      min: number,
      max: number,
    ) => {
      const value = Number(draft[key]);
      if (
        !draft[key].trim() ||
        !Number.isInteger(value) ||
        value < min ||
        value > max
      )
        next[key] = `Enter a whole number between ${min} and ${max}.`;
      return value;
    };
    const timeout = integer("timeout_seconds", 1, 600);
    const concurrency = integer("max_concurrency", 1, 32);
    const context = draft.context_window.trim()
      ? integer("context_window", 1024, 10_000_000)
      : null;
    if (!draft.name.trim()) next.name = "Enter a configuration name.";
    if (!draft.model.trim())
      next.model = "Enter a model ID that actually works.";
    try {
      const url = new URL(draft.base_url);
      if (
        !["http:", "https:"].includes(url.protocol) ||
        url.username ||
        url.password
      )
        throw new Error();
    } catch {
      next.base_url = "Enter a valid model endpoint.";
    }
    if (
      draft.provider_kind !== "ollama" &&
      !draft.api_key.trim() &&
      !saved?.has_api_key
    )
      next.api_key = "Enter an API Key.";
    if (draft.embedding_mode === "custom" && !draft.embedding_model.trim())
      next.embedding_model = "Enter an embedding model ID.";
    if (Object.keys(next).length) {
      setFields(next);
      setAdvanced(true);
      return null;
    }
    return {
      name: draft.name.trim(),
      model: draft.model.trim(),
      provider_kind: draft.provider_kind,
      base_url: draft.base_url.trim(),
      api_key: draft.api_key.trim() || undefined,
      timeout_seconds: timeout,
      max_concurrency: concurrency,
      context_window: context,
      disable_thinking: draft.disable_thinking,
      embedding_provider_kind:
        draft.embedding_mode === "local"
          ? "local_onnx"
          : draft.embedding_provider_kind,
      embedding_model:
        draft.embedding_mode === "local"
          ? "bge-small-zh-v1.5"
          : draft.embedding_model.trim(),
      embedding_base_url:
        draft.embedding_mode === "local"
          ? null
          : draft.embedding_base_url.trim() || null,
      embedding_dimensions: draft.embedding_mode === "local" ? 512 : null,
    };
  }

  async function save(test: boolean) {
    if (flight.current) return;
    const value = input();
    if (!value) return;
    flight.current = true;
    setError(null);
    setPhase("saving");
    try {
      const { provider_kind: providerKind, ...patch } = value;
      const profile =
        saved && saved.provider_kind === providerKind
          ? await models.update(saved.profile_id, {
              ...patch,
              expected_revision: saved.revision,
            })
          : await models.create(value);
      if (!profile) {
        setError(
          "The configuration result is unconfirmed; check it in model settings and retry.",
        );
        return;
      }
      setSaved(profile);
      setDraft((current) => ({ ...current, api_key: "" }));
      if (!test) return;
      setPhase("testing");
      const result = await models.probeProfile(profile.profile_id);
      const current = (await getModelProfiles()).find(
        (item) => item.profile_id === profile.profile_id,
      );
      const passed = probeMatchesProfile(profile, current, result);
      if (current) setSaved(current);
      setProbe({
        state: passed ? "passed" : "failed",
        message: passed
          ? "The model connection test passed; it can be enabled."
          : "The connection failed; check the model, key and recipient.",
      });
    } catch {
      setError(
        "The configuration or test result is not confirmed yet; check and retry.",
      );
    } finally {
      flight.current = false;
      setPhase("idle");
    }
  }

  async function enable() {
    if (flight.current || probe?.state !== "passed" || !saved) return;
    flight.current = true;
    setPhase("enabling");
    setError(null);
    const trace = traceUiAction("memory", "submit");
    try {
      const active = await models.activate(saved.profile_id);
      if (!active?.is_active) throw new Error("model_activation_unconfirmed");
      trace.done("succeeded");
      onEnabled?.();
    } catch {
      trace.done("unknown", { reason: "model_enable_unconfirmed" });
      setError("Enabling the model was not confirmed; check it in settings.");
    } finally {
      flight.current = false;
      setPhase("idle");
    }
  }

  return (
    <OnboardingModelStep
      draft={draft}
      providers={[
        ...PROVIDER_PRESETS.map((preset) => ({
          id: preset.id,
          label: preset.label,
          keyRequired: preset.provider !== "ollama",
          supported: true,
        })),
        {
          id: "custom",
          label: "Custom compatible service",
          keyRequired: true,
          supported: true,
        },
      ]}
      advancedOpen={advanced}
      onAdvancedChange={setAdvanced}
      onDraftChange={edit}
      phase={phase}
      capability={{ saveAndEnable: true }}
      canSaveAndEnable={probe?.state === "passed" && !!saved}
      canTest={!models.pendingProbe && !models.probeStorageFailed}
      canSaveDraft={!models.pendingCreation && !models.creationStorageFailed}
      inputsLocked={flight.current}
      fieldErrors={fields}
      error={error ?? models.error}
      probe={probe}
      profile={
        saved
          ? {
              profile_id: saved.profile_id,
              revision: saved.revision,
              has_configured_api_key:
                saved.has_configured_api_key ?? saved.has_api_key,
              credential_state: saved.credential_state,
            }
          : null
      }
      sourceNames={sourceNames}
      importedConversationCount={importedConversationCount}
      connectionSummary={
        connectionAlreadyApplied
          ? "The selected connection is enabled"
          : "No app connection was chosen this time"
      }
      generationRecipient={draft.base_url || "No recipient filled in yet"}
      embeddingRecipient={
        draft.embedding_mode === "local"
          ? "Built-in local embedding model"
          : draft.embedding_base_url ||
            draft.base_url ||
            "No recipient filled in yet"
      }
      onSaveDraft={() => save(false)}
      onTest={() => save(true)}
      onSaveAndEnable={enable}
      onBack={onBack}
      onSkip={onSkip}
      onActionError={() =>
        setError("The operation is unconfirmed; the draft is kept.")
      }
    />
  );
}
