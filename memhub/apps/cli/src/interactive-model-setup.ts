import type { ModelProfile } from "@one-memory/contracts";
import type { CoreReader } from "./core-client.js";
import type {
  CoreDeleteOperations,
  CoreOperations,
} from "./core-operations.js";
import {
  executeModelProfile,
  type FlowDocument,
  planModelProfile,
} from "./headless-flows.js";
import { approvalFingerprint, listHosts, readHost } from "./host-flow.js";
import { parseSelection, type WizardPrompter } from "./init-wizard.js";
import { executeModelProbe, planModelProbe } from "./model-probe-flow.js";
import {
  executeModelProfileActivate,
  planModelProfileActivate,
} from "./model-write-flows.js";

type ProviderPreset = {
  id: string;
  label: string;
  provider_kind: "openai_compatible" | "ollama";
  base_url: string;
  model: string;
  key_required: boolean;
};

export const CLI_PROVIDER_PRESETS: readonly ProviderPreset[] = [
  {
    id: "openai",
    label: "OpenAI",
    provider_kind: "openai_compatible",
    base_url: "https://api.openai.com/v1",
    model: "gpt-4o-mini",
    key_required: true,
  },
  {
    id: "deepseek",
    label: "DeepSeek",
    provider_kind: "openai_compatible",
    base_url: "https://api.deepseek.com",
    model: "deepseek-flash",
    key_required: true,
  },
  {
    id: "dashscope",
    label: "Alibaba Cloud Model Studio (Qwen)",
    provider_kind: "openai_compatible",
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model: "qwen-flash",
    key_required: true,
  },
  {
    id: "moonshot",
    label: "Kimi (Moonshot)",
    provider_kind: "openai_compatible",
    base_url: "https://api.moonshot.cn/v1",
    model: "kimi-k3",
    key_required: true,
  },
  {
    id: "zhipu",
    label: "Zhipu GLM",
    provider_kind: "openai_compatible",
    base_url: "https://open.bigmodel.cn/api/paas/v4",
    model: "glm-4-flash",
    key_required: true,
  },
  {
    id: "volcengine",
    label: "Volcano Engine Ark (Doubao)",
    provider_kind: "openai_compatible",
    base_url: "https://ark.cn-beijing.volces.com/api/v3",
    model: "doubao-seed-2-1-pro-260628",
    key_required: true,
  },
  {
    id: "hunyuan",
    label: "Tencent Hunyuan",
    provider_kind: "openai_compatible",
    base_url: "https://api.hunyuan.cloud.tencent.com/v1",
    model: "hunyuan-turbos-latest",
    key_required: true,
  },
  {
    id: "siliconflow",
    label: "SiliconFlow",
    provider_kind: "openai_compatible",
    base_url: "https://api.siliconflow.cn/v1",
    model: "deepseek-ai/DeepSeek-V3",
    key_required: true,
  },
  {
    id: "openrouter",
    label: "OpenRouter",
    provider_kind: "openai_compatible",
    base_url: "https://openrouter.ai/api/v1",
    model: "",
    key_required: true,
  },
  {
    id: "ollama",
    label: "Ollama (local)",
    provider_kind: "ollama",
    base_url: "http://127.0.0.1:11434",
    model: "",
    key_required: false,
  },
  {
    id: "custom",
    label: "Custom OpenAI-compatible service",
    provider_kind: "openai_compatible",
    base_url: "",
    model: "",
    key_required: true,
  },
];

export type InteractiveModelSetupState =
  | {
      state: "configured";
      profile: ModelProfile;
      installed_hosts: string[];
    }
  | { state: "skipped" }
  | { state: "cancelled" }
  | { state: "failed"; code: string };

function receiptProfile(document: FlowDocument): ModelProfile | null {
  if (!(document.state === "committed" || document.state === "recovered"))
    return null;
  const value = document.receipt as Partial<ModelProfile>;
  return typeof value.profile_id === "string" &&
    typeof value.revision === "number"
    ? (value as ModelProfile)
    : null;
}

function probePassed(value: unknown): boolean {
  if (value === null || typeof value !== "object") return false;
  const receipt = value as {
    state?: unknown;
    receipt?: { result?: { capabilities?: Record<string, unknown> } };
  };
  const capabilities = receipt.receipt?.result?.capabilities;
  return (
    receipt.state === "completed" &&
    capabilities?.model_available === true &&
    capabilities?.embedding_available === true
  );
}

async function answerWithDefault(
  prompter: WizardPrompter,
  question: string,
  fallback: string,
): Promise<string> {
  const value = (
    await prompter.ask(`${question} (default ${fallback}): `)
  ).trim();
  return value || fallback;
}

async function chooseProvider(
  prompter: WizardPrompter,
): Promise<ProviderPreset | null> {
  prompter.write("Choose a model service:");
  CLI_PROVIDER_PRESETS.forEach((preset, index) => {
    prompter.write(`  ${index + 1}) ${preset.label}`);
  });
  prompter.write("  0) Skip, and only import raw conversations for now");
  while (true) {
    const answer = (
      await prompter.ask("Choose (default DeepSeek, type back to go back): ")
    ).trim();
    if (answer.toLowerCase() === "back") return null;
    if (answer === "0" || answer.toLowerCase() === "skip") return null;
    const index = answer === "" ? 1 : Number(answer) - 1;
    if (
      Number.isSafeInteger(index) &&
      index >= 0 &&
      index < CLI_PROVIDER_PRESETS.length
    ) {
      return CLI_PROVIDER_PRESETS[index]!;
    }
    prompter.write("Unrecognized input. Enter a service number, 0, or back.");
  }
}

async function retryOrSkip(prompter: WizardPrompter): Promise<boolean> {
  while (true) {
    const answer = (
      await prompter.ask(
        "Enter r to fill it in again, or 0 to skip model setup: ",
      )
    )
      .trim()
      .toLowerCase();
    if (answer === "" || answer === "r" || answer === "retry") return true;
    if (answer === "0" || answer === "skip") return false;
    prompter.write("Unrecognized input. Enter r or 0.");
  }
}

async function chooseDetectedHosts(
  prompter: WizardPrompter,
  hostHome: string,
  hostApi: {
    list: typeof listHosts;
    read: typeof readHost;
  },
): Promise<string[]> {
  const listed = await hostApi.list(hostHome);
  if (!("hosts" in listed) || listed.hosts.length === 0) return [];
  const answer = (await prompter.ask("Connect the detected apps? [Y/n]: "))
    .trim()
    .toLowerCase();
  if (["n", "no", "\u5426", "0"].includes(answer)) return [];
  const detected = listed.hosts.filter((host) => host.present);
  if (detected.length === 0) {
    prompter.write(
      "No connectable app was detected, so this was skipped. Run memhub host install later to connect one.",
    );
    return [];
  }
  prompter.write("Detected these apps:");
  detected.forEach((host, index) => {
    prompter.write(`  ${index + 1}) ${host.label}`);
  });
  let selected: number[] | null = null;
  while (selected === null) {
    selected = parseSelection(
      await prompter.ask(
        "Choose the apps to connect (Enter = all, e.g. 1,3): ",
      ),
      detected.length,
    );
    if (selected === null)
      prompter.write("Unrecognized input. Enter the numbers again or all.");
  }
  return selected.flatMap((index) => {
    const host = detected[index];
    return host ? [host.host_id] : [];
  });
}

/** OpenClaw has a layout but the connection client refuses it, so installing
 * it from the wizard always throws. Keep the list aligned with what the
 * client can actually install instead of failing the whole run.
 */
const CLI_INSTALLABLE_HOSTS = new Set(["openclaw"]);

/** Hosts the CLI cannot install but the App can. The summary must never call
 * them "failed": the failure is the CLI's, not the host's. */
const APP_ONLY_HOSTS = new Set(["openclaw"]);

async function installHosts(
  dataDir: string,
  hostIds: readonly string[],
  hostApi: { read: typeof readHost },
): Promise<{ installed: string[]; failed: string[] }> {
  const installed: string[] = [];
  const failed: string[] = [];
  for (const hostId of hostIds) {
    try {
      const preview = await hostApi.read(dataDir, hostId, "previewInstall");
      const fingerprint =
        "result" in preview ? approvalFingerprint(preview.result) : null;
      if (fingerprint === null) {
        failed.push(hostId);
        continue;
      }
      const applied = await hostApi.read(dataDir, hostId, "install", [
        fingerprint,
      ]);
      if ("state" in applied) failed.push(hostId);
      else installed.push(hostId);
    } catch (error) {
      failed.push(hostId);
      if (CLI_INSTALLABLE_HOSTS.has(hostId)) continue;
      if (
        !(error instanceof Error) ||
        !error.message.startsWith("connection_client_")
      ) {
        throw error;
      }
    }
  }
  return { installed, failed };
}

function writeSummary(
  prompter: WizardPrompter,
  profile: ModelProfile,
  installedHosts: readonly string[],
  failedHosts: readonly { hostId: string; label: string }[],
): void {
  prompter.write("Configuration summary:");
  prompter.write(`  Model: ${profile.name} / ${profile.model}`);
  prompter.write(
    `  App connections: ${installedHosts.length > 0 ? installedHosts.join(", ") : "Skipped"}`,
  );
  for (const failure of failedHosts) {
    if (APP_ONLY_HOSTS.has(failure.hostId)) {
      prompter.write(`  Connect in the app: ${failure.label}`);
      continue;
    }
    prompter.write(`  Connect failed: ${failure.label}`);
  }
  prompter.write(
    "Next step: run memhub status to see the state, or memhub recall <question> to search memories.",
  );
}

export async function runInteractiveModelSetup(
  prompter: WizardPrompter,
  clients: {
    reader: CoreReader;
    operations: CoreOperations & CoreDeleteOperations;
  },
  options: {
    dataDir: string;
    hostHome: string;
    installHosts: boolean;
    hostApi?: { list: typeof listHosts; read: typeof readHost };
  },
): Promise<InteractiveModelSetupState> {
  if (!prompter.askSecret) return { state: "skipped" };
  while (true) {
    const preset = await chooseProvider(prompter);
    if (preset === null) {
      prompter.write("Model setup skipped; only raw import is kept.");
      return { state: "skipped" };
    }
    const baseUrl = await answerWithDefault(
      prompter,
      "Model service address",
      preset.base_url,
    );
    if (baseUrl.toLowerCase() === "back") continue;
    const model = await answerWithDefault(prompter, "Model name", preset.model);
    if (model.toLowerCase() === "back") continue;
    if (!model) {
      prompter.write("The model name can't be empty. Choose again.");
      continue;
    }
    const apiKey = preset.key_required
      ? await prompter.askSecret("API Key (input isn't echoed): ")
      : undefined;
    if (apiKey?.toLowerCase() === "back") continue;
    if (preset.key_required && !apiKey) {
      prompter.write(
        "No API Key entered. Enter back to go back, or choose a model service again.",
      );
      continue;
    }
    const hostApi = options.hostApi ?? { list: listHosts, read: readHost };
    const selectedHosts = options.installHosts
      ? await chooseDetectedHosts(prompter, options.hostHome, hostApi)
      : [];
    const hostLabels = options.installHosts
      ? await hostApi.list(options.hostHome).then((listed) => {
          if (!("hosts" in listed)) return new Map<string, string>();
          return new Map(
            listed.hosts.map((host) => [host.host_id, host.label]),
          );
        })
      : new Map<string, string>();
    const planned = planModelProfile({
      name: preset.label,
      provider_kind: preset.provider_kind,
      base_url: baseUrl,
      model,
      embedding_provider_kind: "local_onnx",
      embedding_model: "bge-small-zh-v1.5",
      embedding_dimensions: 512,
      timeout_seconds: 90,
      max_concurrency: 4,
      disable_thinking: true,
    });
    if (
      planned.state !== "preview" ||
      planned.plan.kind !== "model_profile_create"
    ) {
      return { state: "failed", code: "model_config_invalid" };
    }
    const created = await executeModelProfile(
      planned.plan,
      apiKey,
      clients.reader,
      clients.operations,
    );
    const profile = receiptProfile(created);
    if (profile === null) {
      prompter.write("Model configuration wasn't saved.");
      if (await retryOrSkip(prompter)) continue;
      return { state: "skipped" };
    }

    try {
      const probePlan = await planModelProbe(
        clients.reader,
        profile.profile_id,
      );
      const probed =
        probePlan.state === "preview"
          ? await executeModelProbe(
              probePlan.plan,
              clients.reader,
              clients.operations,
            )
          : probePlan;
      if (!probePassed(probed)) {
        await clients.operations.delete(
          `/api/v1/model-profiles/${encodeURIComponent(profile.profile_id)}`,
        );
        prompter.write(
          "The model connection didn't pass. Fill it in again, or enter 0 to skip the rest of the setup.",
        );
        if (await retryOrSkip(prompter)) continue;
        prompter.write("Model setup skipped; only raw import is kept.");
        return { state: "skipped" };
      }
    } catch (error) {
      await clients.operations
        .delete(
          `/api/v1/model-profiles/${encodeURIComponent(profile.profile_id)}`,
        )
        .catch(() => undefined);
      throw error;
    }

    const activation = planModelProfileActivate({
      profileId: profile.profile_id,
    });
    if (
      activation.state !== "preview" ||
      activation.plan.kind !== "model_profile_activate"
    ) {
      return { state: "failed", code: "model_activate_plan_invalid" };
    }
    const activated = await executeModelProfileActivate(
      activation.plan,
      clients.operations,
    );
    if (activated.state !== "committed") {
      return { state: "failed", code: "model_activate_failed" };
    }

    const { installed, failed } = await installHosts(
      options.dataDir,
      selectedHosts,
      hostApi,
    );
    writeSummary(
      prompter,
      profile,
      installed,
      failed.map((hostId) => ({
        hostId,
        label: hostLabels.get(hostId) ?? hostId,
      })),
    );
    return {
      state: "configured",
      profile,
      installed_hosts: installed,
    };
  }
}
