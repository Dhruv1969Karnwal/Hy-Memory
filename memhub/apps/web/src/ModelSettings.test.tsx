// @vitest-environment jsdom

import type { ModelProbeResult, ModelProfile } from "@one-memory/contracts";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createRef } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  ModelSettings,
  type ModelSettingsActions,
  probeLabel,
} from "./ModelSettings";
import { PROVIDER_PRESETS } from "./provider-presets";

const create = vi.fn();
const probeProfile = vi.fn();
const activate = vi.fn();
const remove = vi.fn();
const update = vi.fn();

it("explains an insufficient model balance with the required recovery", () => {
  expect(probeLabel("insufficient_balance")).toBe(
    "The model account balance is exhausted; top up or change the key and test again, processing resumes automatically",
  );
});

let profiles: ModelProfile[];
let probes: Record<string, ModelProbeResult> = {};

const profile: ModelProfile = {
  profile_id: "profile-1",
  name: "Saved provider",
  provider_kind: "openai_compatible",
  base_url: "https://models.example.com/v1",
  model: "memory-model",
  embedding_model: "bge-small-zh-v1.5",
  embedding_provider_kind: "local_onnx",
  embedding_base_url: null,
  embedding_dimensions: 512,
  has_api_key: true,
  timeout_seconds: 60,
  max_concurrency: 1,
  context_window: null,
  disable_thinking: false,
  capabilities: {
    endpoint_reachable: false,
    model_available: false,
    structured_output: false,
    embedding_available: false,
  },
  revision: 1,
  is_active: false,
  created_at: "2026-09-01T00:00:00Z",
  updated_at: "2026-09-01T00:00:00Z",
};

/** A downstream distribution preset: everything filled but the key. */
const preset: ModelProfile = {
  ...profile,
  profile_id: "preset-1",
  name: "Preconfigured model service",
  base_url: "https://models.example.com/v1",
  model: "memory-model",
  has_api_key: false,
  disable_thinking: true,
};

vi.mock("./use-model-profiles", () => ({
  useModelProfiles: () => ({
    profiles,
    probe: null,
    probeFor: (value: ModelProfile) =>
      probes[value.profile_id]?.profile_revision === value.revision
        ? probes[value.profile_id]
        : null,
    probeErrorFor: () => null,
    busy: null,
    error: null,
    create,
    probeProfile,
    activate,
    remove,
    update,
  }),
}));

describe("ModelSettings", () => {
  it("exposes distinct edit, add and test actions from the same owner without activating", async () => {
    const controls = createRef<ModelSettingsActions>();
    render(<ModelSettings controlsRef={controls} />);
    act(() => {
      expect(controls.current?.edit(profile.profile_id)).toBe(true);
    });
    expect(
      (screen.getByLabelText("Configuration name") as HTMLInputElement).value,
    ).toBe(profile.name);
    act(() => {
      expect(controls.current?.add()).toBe(true);
    });
    expect(
      (screen.getByLabelText("Configuration name") as HTMLInputElement).value,
    ).toBe("");
    await act(async () => {
      await controls.current?.test(profile.profile_id);
    });
    expect(probeProfile).toHaveBeenCalledExactlyOnceWith(profile.profile_id);
    expect(activate).not.toHaveBeenCalled();
    expect(create).not.toHaveBeenCalled();
    expect(update).not.toHaveBeenCalled();
    expect(controls.current?.edit("missing")).toBe(false);
    await expect(controls.current?.test("missing")).rejects.toThrow(
      "model_profile_not_available",
    );
  });
  afterEach(cleanup);

  beforeEach(() => {
    vi.clearAllMocks();
    probes = {};
    profiles = [profile];
    create.mockResolvedValue(profile);
    update.mockResolvedValue({ ...preset, has_api_key: true, revision: 2 });
  });

  it("submits a generation-only profile and leaves embedding to the builtin", async () => {
    const user = userEvent.setup();
    render(<ModelSettings />);

    expect((screen.getByLabelText(/^API Key/) as HTMLInputElement).type).toBe(
      "password",
    );
    // The default view should not contain embedding. It is a built-in implementation detail users cannot change;
    // a read-only input here would only make readers think they missed some configuration.
    expect(screen.queryByLabelText("Embedding")).toBeNull();
    expect(screen.queryByText(/Built-in bge-small-zh-v1.5/)).toBeNull();
    await user.type(screen.getByLabelText("Configuration name"), "My provider");
    await user.type(screen.getByLabelText("Generation model name"), "model-v1");
    await user.type(screen.getByLabelText(/^API Key/), "secret-value");
    await user.click(
      screen.getByRole("button", { name: "Save configuration" }),
    );

    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "My provider",
        provider_kind: "openai_compatible",
        model: "model-v1",
        api_key: "secret-value",
      }),
    );
    expect(create.mock.calls[0][0].embedding_model).toBeUndefined();
    expect(
      screen.getByText(/You only need to configure the generation model/),
    ).toBeTruthy();
  });

  it("edits an existing custom profile without creating another UUID or refilling the stored key", async () => {
    update.mockResolvedValue(profile);
    render(<ModelSettings />);
    await userEvent.click(
      screen.getByRole("button", { name: "Edit configuration" }),
    );
    expect(screen.getByLabelText("Generation model name")).toHaveProperty(
      "value",
      profile.model,
    );
    expect(screen.getByLabelText("Base URL")).toHaveProperty(
      "value",
      profile.base_url,
    );
    expect(screen.getByLabelText("Configuration name")).toHaveProperty(
      "value",
      profile.name,
    );
    expect(screen.getByLabelText(/^API Key/)).toHaveProperty("value", "");
    expect(screen.getByText(/Leave blank to keep the saved key/)).toBeTruthy();
    await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
    expect(update).toHaveBeenCalledExactlyOnceWith(profile.profile_id, {
      expected_revision: 1,
    });
    expect(create).not.toHaveBeenCalled();
    expect(probeProfile).not.toHaveBeenCalled();
    expect(activate).not.toHaveBeenCalled();
    expect(
      screen.getByRole("article", { name: "Saved model configurations" })
        .textContent,
    ).toContain("Revision 1");
  });

  it("offers the same vendor choices while editing and does not reuse the saved key", async () => {
    const user = userEvent.setup();
    profiles = [
      {
        ...profile,
        name: "DeepSeek",
        base_url: "https://api.deepseek.com",
        model: "deepseek-flash",
        has_api_key: true,
      },
    ];
    render(<ModelSettings />);
    await user.click(
      screen.getByRole("button", { name: "Edit configuration" }),
    );
    const vendor = screen.getByLabelText("Provider preset");
    expect(vendor.textContent).toContain("DeepSeek");
    await user.click(vendor);
    expect(
      within(screen.getByRole("listbox"))
        .getAllByRole("option")
        .map((item) => item.textContent),
    ).toEqual([
      ...PROVIDER_PRESETS.map((item) => item.label),
      "Custom compatible service",
    ]);
    await user.click(screen.getByRole("option", { name: "Kimi (Moonshot)" }));
    expect(screen.getByLabelText("Base URL")).toHaveProperty(
      "value",
      "https://api.moonshot.cn/v1",
    );
    expect(screen.getByLabelText("Generation model name")).toHaveProperty(
      "value",
      "kimi-k3",
    );
    expect(screen.getByLabelText("Configuration name")).toHaveProperty(
      "value",
      "Kimi (Moonshot)",
    );
    expect(screen.getByText(/carried to the new address/)).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    expect(update).not.toHaveBeenCalled();
    await user.type(screen.getByLabelText(/^API Key/), "new-key");
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    expect(update).toHaveBeenCalledExactlyOnceWith(profile.profile_id, {
      expected_revision: 1,
      name: "Kimi (Moonshot)",
      model: "kimi-k3",
      base_url: "https://api.moonshot.cn/v1",
      api_key: "new-key",
      disable_thinking: true,
    });
    expect(create).not.toHaveBeenCalled();
  });

  it("starts a new profile when an edit switches to a different protocol", async () => {
    const user = userEvent.setup();
    profiles = [
      {
        ...profile,
        name: "DeepSeek",
        base_url: "https://api.deepseek.com",
        model: "deepseek-flash",
        has_api_key: true,
      },
    ];
    render(<ModelSettings />);
    await user.click(
      screen.getByRole("button", { name: "Edit configuration" }),
    );
    await user.click(screen.getByLabelText("Provider preset"));
    await user.click(
      within(screen.getByRole("listbox")).getByRole("option", {
        name: "Ollama (local)",
      }),
    );
    await user.type(screen.getByLabelText("Generation model name"), "qwen2.5");
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    expect(update).not.toHaveBeenCalled();
    expect(create).toHaveBeenCalledExactlyOnceWith({
      name: "Ollama (local)",
      provider_kind: "ollama",
      model: "qwen2.5",
      base_url: "http://127.0.0.1:11434",
      disable_thinking: true,
    });
  });

  it("preserves edited inputs and replacement key on failure then saves the same identity", async () => {
    update.mockResolvedValueOnce(undefined).mockResolvedValueOnce({
      ...profile,
      revision: 2,
      model: "changed",
      is_active: false,
    });
    render(<ModelSettings />);
    await userEvent.click(
      screen.getByRole("button", { name: "Edit configuration" }),
    );
    await userEvent.clear(screen.getByLabelText("Generation model name"));
    await userEvent.type(
      screen.getByLabelText("Generation model name"),
      "changed",
    );
    await userEvent.type(
      screen.getByLabelText(/^API Key/),
      "synthetic-replacement",
    );
    await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
    expect(screen.getByLabelText("Generation model name")).toHaveProperty(
      "value",
      "changed",
    );
    expect(screen.getByLabelText(/^API Key/)).toHaveProperty(
      "value",
      "synthetic-replacement",
    );
    expect(screen.getByRole("alert").textContent).not.toContain(
      "synthetic-replacement",
    );
    await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
    expect(update).toHaveBeenLastCalledWith(profile.profile_id, {
      expected_revision: 1,
      model: "changed",
      api_key: "synthetic-replacement",
    });
    expect(create).not.toHaveBeenCalled();
    expect(activate).not.toHaveBeenCalled();
    expect(probeProfile).not.toHaveBeenCalled();
    expect(
      screen.getByRole("article", { name: "Saved model configurations" })
        .textContent,
    ).toContain("Revision 2");
    expect(screen.queryByLabelText(/^API Key/)).toBeNull();
  });

  it("keeps unrelated model fields when editing an existing advanced embedding configuration", async () => {
    const existing = {
      ...profile,
      embedding_model: "local-embedding",
      embedding_provider_kind: "ollama" as const,
      embedding_base_url: "http://127.0.0.1:11434",
      context_window: 32000,
      timeout_seconds: 90,
      max_concurrency: 2,
    };
    profiles = [existing];
    update.mockResolvedValue(existing);
    render(<ModelSettings />);
    await userEvent.click(
      screen.getByRole("button", { name: "Edit configuration" }),
    );
    expect(screen.getByLabelText("Embedding model name")).toHaveProperty(
      "value",
      "local-embedding",
    );
    await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
    expect(update).toHaveBeenCalledExactlyOnceWith(profile.profile_id, {
      expected_revision: 1,
    });
  });

  it("does not replace an edit draft with a newer external profile revision", async () => {
    const view = render(<ModelSettings />);
    await userEvent.click(
      screen.getByRole("button", { name: "Edit configuration" }),
    );
    await userEvent.type(
      screen.getByLabelText("Generation model name"),
      "-draft",
    );
    profiles = [{ ...profile, revision: 2, model: "external" }];
    view.rerender(<ModelSettings />);
    expect(screen.getByLabelText("Generation model name")).toHaveProperty(
      "value",
      "memory-model-draft",
    );
    expect(screen.getByRole("button", { name: "Save changes" })).toHaveProperty(
      "disabled",
      true,
    );
    expect(
      screen.getByText(/The configuration changed elsewhere/),
    ).toBeTruthy();
    expect(update).not.toHaveBeenCalled();
  });

  it("preserves a new custom embedding when its disclosure is collapsed", async () => {
    render(<ModelSettings />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Generation model name"), "model-v1");
    await user.type(screen.getByLabelText(/^API Key/), "synthetic-key");
    const disclosure = screen.getByRole("button", {
      name: "Advanced: configure the embedding model yourself",
    });
    await user.click(disclosure);
    await user.selectOptions(
      screen.getByLabelText("Embedding configuration mode"),
      "custom",
    );
    await user.type(
      screen.getByLabelText("Embedding model name"),
      "custom-vector",
    );
    await user.type(
      screen.getByLabelText("Embedding model Base URL"),
      "https://api.openai.com/v1",
    );
    await user.click(disclosure);
    await user.click(
      screen.getByRole("button", { name: "Save configuration" }),
    );
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        embedding_provider_kind: "openai_compatible",
        embedding_model: "custom-vector",
        embedding_base_url: "https://api.openai.com/v1",
      }),
    );
    expect(probeProfile).not.toHaveBeenCalled();
    expect(activate).not.toHaveBeenCalled();
  });

  it("does not turn opening advanced fields into an embedding mode change", async () => {
    render(<ModelSettings />);
    await userEvent.type(
      screen.getByLabelText("Generation model name"),
      "model-v1",
    );
    await userEvent.type(screen.getByLabelText(/^API Key/), "synthetic-key");
    await userEvent.click(
      screen.getByRole("button", {
        name: "Advanced: configure the embedding model yourself",
      }),
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Save configuration" }),
    );
    expect(create).toHaveBeenCalledOnce();
    expect(create.mock.calls[0][0].embedding_model).toBeUndefined();
  });

  it("saves an edited custom embedding after collapsing its fields", async () => {
    profiles = [
      {
        ...profile,
        embedding_provider_kind: "ollama",
        embedding_model: "vector-old",
        embedding_base_url: "http://127.0.0.1:11434",
      },
    ];
    render(<ModelSettings />);
    await userEvent.click(
      screen.getByRole("button", { name: "Edit configuration" }),
    );
    await userEvent.clear(screen.getByLabelText("Embedding model name"));
    await userEvent.type(
      screen.getByLabelText("Embedding model name"),
      "vector-new",
    );
    await userEvent.click(
      screen.getByRole("button", {
        name: "Advanced: configure the embedding model yourself",
      }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
    expect(update).toHaveBeenCalledWith(profile.profile_id, {
      expected_revision: 1,
      embedding_model: "vector-new",
    });
  });

  it("explicitly switches an edited remote embedding to builtin without retaining its dimensions", async () => {
    profiles = [
      {
        ...profile,
        embedding_provider_kind: "ollama",
        embedding_model: "vector-old",
        embedding_base_url: "http://127.0.0.1:11434",
        embedding_dimensions: 768,
      },
    ];
    render(<ModelSettings />);
    await userEvent.click(
      screen.getByRole("button", { name: "Edit configuration" }),
    );
    await userEvent.selectOptions(
      screen.getByLabelText("Embedding configuration mode"),
      "local",
    );
    await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
    expect(update).toHaveBeenCalledWith(profile.profile_id, {
      expected_revision: 1,
      embedding_provider_kind: "local_onnx",
      embedding_model: "bge-small-zh-v1.5",
      embedding_dimensions: 512,
    });
  });

  it("reopens invalid custom fields instead of saving builtin after collapse", async () => {
    render(<ModelSettings />);
    await userEvent.type(
      screen.getByLabelText("Generation model name"),
      "model-v1",
    );
    await userEvent.type(screen.getByLabelText(/^API Key/), "synthetic-key");
    const disclosure = screen.getByRole("button", {
      name: "Advanced: configure the embedding model yourself",
    });
    await userEvent.click(disclosure);
    await userEvent.selectOptions(
      screen.getByLabelText("Embedding configuration mode"),
      "custom",
    );
    await userEvent.click(disclosure);
    await userEvent.click(
      screen.getByRole("button", { name: "Save configuration" }),
    );
    expect(create).not.toHaveBeenCalled();
    expect(disclosure.getAttribute("aria-expanded")).toBe("true");
    expect(
      screen
        .getByLabelText(/^Embedding model name/)
        .getAttribute("aria-invalid"),
    ).toBe("true");
  });

  it("attaches test feedback to the named profile revision and not a different saved card", async () => {
    profiles = [
      profile,
      {
        ...profile,
        profile_id: "second",
        name: "Second",
        base_url: "https://second.example/v1",
      },
    ];
    probes = {
      [profile.profile_id]: {
        profile_id: profile.profile_id,
        profile_revision: 1,
        capabilities: profile.capabilities,
        stages: [
          {
            stage: "authentication",
            ok: false,
            code: "unauthorized",
            latency_ms: 1,
          },
        ],
      },
    };
    const view = render(<ModelSettings />);
    const result = screen.getByRole("region", {
      name: "Saved provider revision 1 test result",
    });
    expect(result.textContent).toContain("Authentication failed");
    expect(result.textContent).toContain("https://models.example.com/v1");
    expect(
      screen.queryByRole("region", { name: "Second revision 1 test result" }),
    ).toBeNull();
    profiles = [{ ...profile, revision: 2 }, profiles[1]];
    view.rerender(<ModelSettings />);
    expect(screen.queryByText(/Authentication failed/)).toBeNull();
  });

  it("does not enable activation from older capability flags after this revision's failed test", () => {
    profiles = [
      {
        ...profile,
        capabilities: {
          ...profile.capabilities,
          endpoint_reachable: true,
          model_available: true,
          embedding_available: true,
        },
      },
    ];
    probes = {
      [profile.profile_id]: {
        profile_id: profile.profile_id,
        profile_revision: 1,
        capabilities: profile.capabilities,
        stages: [
          {
            stage: "authentication",
            ok: false,
            code: "unauthorized",
            latency_ms: 1,
          },
        ],
      },
    };
    render(<ModelSettings />);
    expect(
      screen.getByRole("button", { name: "Set as current model" }),
    ).toHaveProperty("disabled", true);
    expect(
      screen.getByRole("region", {
        name: "Saved provider revision 1 test result",
      }).textContent,
    ).toContain("Authentication failed");
    expect(activate).not.toHaveBeenCalled();
  });

  it("distinguishes a configured but unreadable key without telling the user the whole config changed", () => {
    profiles = [
      {
        ...profile,
        has_api_key: false,
        has_configured_api_key: true,
        credential_state: "unavailable",
      },
    ];
    render(<ModelSettings />);
    expect(
      screen.getByText(/MemHub needs access to the keychain/),
    ).toBeTruthy();
    expect(screen.getByText(/Always Allow/)).toBeTruthy();
    expect(
      screen.queryByText(
        /This version cannot read old keychain credentials yet/,
      ),
    ).toBeNull();
    expect(screen.queryByText(/Waiting for a key/)).toBeNull();
  });

  it("does not render a stored key and exposes explicit activation", async () => {
    profiles = [
      {
        ...profile,
        capabilities: {
          ...profile.capabilities,
          endpoint_reachable: true,
          model_available: true,
          embedding_available: true,
        },
      },
    ];
    const user = userEvent.setup();
    const { container } = render(<ModelSettings />);

    expect(container.textContent).not.toContain("secret-value");
    await user.click(
      screen.getByRole("button", { name: "Set as current model" }),
    );
    expect(activate).toHaveBeenCalledWith("profile-1");
  });

  it("hands a successful save to its card, clears only the key, and never tests or activates", async () => {
    profiles = [];
    const user = userEvent.setup();
    render(<ModelSettings />);
    await user.type(
      screen.getByLabelText("Generation model name"),
      "memory-model",
    );
    await user.type(screen.getByLabelText(/^API Key/), "test-only-key");
    await user.click(
      screen.getByRole("button", { name: "Save configuration" }),
    );
    const card = screen.getByRole("article", {
      name: "Saved model configurations",
    });
    expect(card.textContent).toContain("Saved, pending test");
    expect(card.textContent).toContain("models.example.com");
    expect(document.activeElement).toBe(card);
    expect(
      screen.queryByRole("button", { name: "Save configuration" }),
    ).toBeNull();
    expect(probeProfile).not.toHaveBeenCalled();
    expect(activate).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Add configuration" }));
    expect(screen.getByLabelText(/^API Key/)).toHaveProperty("value", "");
    expect(create).toHaveBeenCalledTimes(1);
  });

  it("suggests an editable name without changing the endpoint or model", async () => {
    const user = userEvent.setup();
    render(<ModelSettings />);
    await user.type(screen.getByLabelText("Generation model name"), "model-v1");
    expect(screen.getByLabelText("Configuration name")).toHaveProperty(
      "value",
      "OpenAI-compatible · model-v1",
    );
    await user.clear(screen.getByLabelText("Configuration name"));
    await user.type(screen.getByLabelText("Configuration name"), "My choice");
    await user.type(screen.getByLabelText("Generation model name"), "-v2");
    await user.clear(screen.getByLabelText("Base URL"));
    await user.type(
      screen.getByLabelText("Base URL"),
      "https://gateway.example/v1",
    );
    await user.selectOptions(screen.getByLabelText("Protocol"), "ollama");
    expect(screen.getByLabelText("Configuration name")).toHaveProperty(
      "value",
      "My choice",
    );
    expect(screen.getByLabelText("Generation model name")).toHaveProperty(
      "value",
      "model-v1-v2",
    );
    expect(screen.getByLabelText("Base URL")).toHaveProperty(
      "value",
      "https://gateway.example/v1",
    );
  });

  it("shows field-local Chinese required errors for mouse and keyboard submission", async () => {
    const user = userEvent.setup();
    render(<ModelSettings />);
    await user.click(
      screen.getByRole("button", { name: "Save configuration" }),
    );
    expect(screen.getByText("Enter the generation model name.")).toBeTruthy();
    expect(screen.getByText("Enter an API Key.")).toBeTruthy();
    expect(
      screen
        .getByLabelText("Generation model name")
        .getAttribute("aria-invalid"),
    ).toBe("true");
    expect(document.activeElement).toBe(
      screen.getByLabelText("Generation model name"),
    );
    await user.keyboard("{Enter}");
    expect(create).not.toHaveBeenCalled();
  });

  it.each([
    "https://key.example/private-value",
    "www.key.example/private-value",
  ])(
    "rejects URL-shaped keys locally without echoing the value (%#)",
    async (key) => {
      const user = userEvent.setup();
      const { container } = render(<ModelSettings />);
      await user.type(
        screen.getByLabelText("Generation model name"),
        "model-v1",
      );
      await user.type(screen.getByLabelText(/^API Key/), key);
      await user.click(
        screen.getByRole("button", { name: "Save configuration" }),
      );
      expect(
        screen.getByText(
          "This needs an API Key, not a URL. Enter the key provided by the provider.",
        ),
      ).toBeTruthy();
      expect(container.textContent).not.toContain(key);
      expect(screen.getByLabelText(/^API Key/)).toHaveProperty(
        "type",
        "password",
      );
      expect(screen.getByLabelText(/^API Key/)).toHaveProperty("value", key);
      expect(create).not.toHaveBeenCalled();
    },
  );

  it("preserves failed input and single-flights rapid submits", async () => {
    let resolve!: (value: ModelProfile | undefined) => void;
    create.mockImplementationOnce(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    const user = userEvent.setup();
    render(<ModelSettings />);
    await user.type(screen.getByLabelText("Generation model name"), "model-v1");
    await user.type(screen.getByLabelText(/^API Key/), "test-only-key");
    const form = screen
      .getByRole("button", { name: "Save configuration" })
      .closest("form");
    if (!form) throw new Error("Missing model form");
    fireEvent.submit(form);
    fireEvent.submit(form);
    expect(create).toHaveBeenCalledTimes(1);
    await act(async () => resolve(undefined));
    expect(screen.getByLabelText("Generation model name")).toHaveProperty(
      "value",
      "model-v1",
    );
    expect(screen.getByLabelText(/^API Key/)).toHaveProperty(
      "value",
      "test-only-key",
    );
    await user.click(
      screen.getByRole("button", { name: "Save configuration" }),
    );
    expect(create).toHaveBeenCalledTimes(2);
  });

  it("rejects invalid endpoint input without changing it", async () => {
    const user = userEvent.setup();
    render(<ModelSettings />);
    await user.type(screen.getByLabelText("Generation model name"), "model-v1");
    await user.type(screen.getByLabelText(/^API Key/), "test-only-key");
    await user.clear(screen.getByLabelText("Base URL"));
    await user.type(screen.getByLabelText("Base URL"), "not-an-endpoint");
    await user.click(
      screen.getByRole("button", { name: "Save configuration" }),
    );
    expect(
      screen.getByText("Enter a complete http:// or https:// endpoint."),
    ).toBeTruthy();
    expect(screen.getByLabelText("Base URL")).toHaveProperty(
      "value",
      "not-an-endpoint",
    );
    expect(create).not.toHaveBeenCalled();
  });

  it("exposes an advanced path to pair a remote generator with local Ollama embeddings", async () => {
    const user = userEvent.setup();
    render(<ModelSettings />);

    // The capability is still there, just tucked into the collapsible section: hidden by default, shown when expanded.
    expect(screen.queryByLabelText("Embedding model name")).toBeNull();
    await user.click(
      screen.getByRole("button", {
        name: "Advanced: configure the embedding model yourself",
      }),
    );
    await user.selectOptions(
      screen.getByLabelText("Embedding configuration mode"),
      "custom",
    );
    await user.type(
      screen.getByLabelText("Configuration name"),
      "Remote with local embedding",
    );
    await user.type(
      screen.getByLabelText("Generation model name"),
      "remote-model",
    );
    await user.type(
      screen.getByLabelText("Embedding model name"),
      "nomic-embed-text",
    );
    await user.selectOptions(
      screen.getByLabelText("Embedding model source"),
      "ollama",
    );
    await user.type(
      screen.getByLabelText("Embedding model Base URL"),
      "http://127.0.0.1:11434",
    );
    await user.type(screen.getByLabelText(/^API Key/), "secret-value");
    await user.click(
      screen.getByRole("button", { name: "Save configuration" }),
    );

    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        embedding_provider_kind: "ollama",
        embedding_base_url: "http://127.0.0.1:11434",
      }),
    );
  });

  it("Protocols the backend cannot activate never appear in the dropdown", () => {
    render(<ModelSettings />);

    const providerSelect = screen.getByLabelText(
      "Protocol",
    ) as HTMLSelectElement;
    const labels = Array.from(providerSelect.options).map(
      (option) => option.textContent ?? "",
    );

    expect(labels.some((text) => text.startsWith("OpenAI-compatible"))).toBe(
      true,
    );
    expect(labels.some((text) => text.startsWith("Ollama (local)"))).toBe(true);

    // ModelProfileStore.activate() throws for these two providers:
    // model_provider_not_supported. They used to stay in the dropdown as disabled options,
    // which put two dead ends in plain sight and made people parse the "not supported yet" in parentheses.
    expect(labels.some((text) => text.includes("Anthropic"))).toBe(false);
    expect(labels.some((text) => text.includes("HY Memory"))).toBe(false);
    expect(providerSelect.options.length).toBe(2);
  });

  it("Saved configurations do not print the built-in embedding model", () => {
    profiles = [profile];
    render(<ModelSettings />);

    // Core backfills bge-small-zh-v1.5 for every configuration, so this line used to print something
    // that users neither chose nor can change.
    expect(screen.queryByText(/bge-small-zh-v1\.5/)).toBeNull();
    expect(screen.getByText(/memory-model/)).toBeTruthy();
  });

  it("But embedding models the user configured must be printed", () => {
    profiles = [
      {
        ...profile,
        embedding_model: "nomic-embed-text",
        embedding_provider_kind: "ollama",
        embedding_base_url: "http://127.0.0.1:11434",
      },
    ];
    render(<ModelSettings />);

    expect(screen.getByText(/nomic-embed-text/)).toBeTruthy();
  });

  it("With a preset profile, offer the paste-a-key path first and collapse the custom form", () => {
    profiles = [preset];
    render(<ModelSettings />);

    expect(
      screen.getByRole("region", { name: "Preconfigured model service" }),
    ).toBeTruthy();
    // When pasting a key is all it takes, there is no reason to confront people with protocol, Base URL and timeouts first.
    expect(screen.queryByLabelText("Protocol")).toBeNull();
    expect(
      screen.getByRole("button", { name: "Add a custom model" }),
    ).toBeTruthy();
  });

  it("Without a preset profile the form opens directly, with no extra click", () => {
    profiles = [];
    render(<ModelSettings />);

    expect(screen.getByLabelText("Protocol")).toBeTruthy();
  });

  it("no longer promises an HY default model that will not arrive", () => {
    render(<ModelSettings />);
    expect(screen.queryByText(/not open yet/)).toBeNull();
  });

  describe("Preset profile", () => {
    it("Asks for one key only; endpoint and model name stay out of sight", async () => {
      profiles = [preset];
      render(<ModelSettings />);

      const card = screen.getByRole("region", {
        name: "Preconfigured model service",
      });
      expect(card.textContent).toContain("Paste API Key");
      // Users do not need to know what a Base URL is or which model name to write:
      // showing those two things only makes them think they still have something to do.
      expect(card.textContent).not.toContain("models.example.com");
      expect(card.textContent).not.toContain("memory-model");
    });

    it("When the key is empty it says waiting for a key, not a configuration error", async () => {
      // "Waiting for you to paste a key" and "you filled something in wrong" are different things. Mixed into one sentence, users go
      // rechecking things they got right while the step that actually matters never comes up.
      profiles = [preset];
      render(<ModelSettings />);

      const card = screen.getByRole("region", {
        name: "Preconfigured model service",
      });
      expect(card.textContent).toContain("Waiting for a key");
      expect(card.textContent).not.toContain("Configuration error");
      expect(card.textContent).not.toContain("Invalid configuration");
    });

    it("Provides a clickable way to apply for a key", async () => {
      profiles = [preset];
      render(<ModelSettings />);

      const link = screen.getByRole("link", { name: /Get one/ });
      expect(link.getAttribute("href")).toBe("https://models.example.com");
    });

    it("Paste the key and go; nothing else to fill in", async () => {
      profiles = [preset];
      const user = userEvent.setup();
      render(<ModelSettings />);

      await user.type(screen.getByLabelText("Paste API Key"), "preset-key");
      await user.click(screen.getByRole("button", { name: "Save key" }));

      expect(update).toHaveBeenCalledWith("preset-1", {
        expected_revision: 1,
        api_key: "preset-key",
      });
      expect(probeProfile).not.toHaveBeenCalled();
      expect(activate).not.toHaveBeenCalled();
    });

    it("also rejects a URL pasted into a preset key without a request", async () => {
      profiles = [preset];
      const user = userEvent.setup();
      render(<ModelSettings />);
      await user.type(
        screen.getByLabelText("Paste API Key"),
        "https://keys.example/path",
      );
      await user.click(screen.getByRole("button", { name: "Save key" }));
      expect(
        screen.getByText(
          "This needs an API Key, not a URL. Enter the key provided by the provider.",
        ),
      ).toBeTruthy();
      expect(update).not.toHaveBeenCalled();
    });

    it("A preset profile present must not make customization harder", async () => {
      const user = userEvent.setup();
      profiles = [preset];
      render(<ModelSettings />);

      // No ordering dependencies like "delete the preset before creating one" are allowed. The form starts collapsed,
      // but collapsed is not an ordering dependency: one click opens it without touching the preset.
      await user.click(
        screen.getByRole("button", { name: "Add a custom model" }),
      );

      expect(screen.getByLabelText("Configuration name")).toBeTruthy();
      expect(
        screen.getByRole("button", { name: "Save configuration" }),
      ).toHaveProperty("disabled", false);
      expect(
        screen.getByRole("button", { name: "Delete this card" }),
      ).toBeTruthy();
    });

    it("A regular profile is not rendered as a preset", () => {
      // The assertion is that there is no preset card named after it, not that some phrase is absent:
      // the latter never shows up once this profile has a key, so it would pass without testing anything.
      render(<ModelSettings />);

      expect(
        screen.queryByRole("region", { name: "Saved provider" }),
      ).toBeNull();
      expect(
        screen.getByRole("button", { name: "Test connection" }),
      ).toBeTruthy();
    });
  });
});

describe("Setting a preset card as the current model", () => {
  afterEach(cleanup);

  beforeEach(() => {
    vi.clearAllMocks();
    // How it looks after pasting the key: the key is there, but nothing was probed, so all capability flags are false.
    profiles = [{ ...preset, has_api_key: true, revision: 2 }];
    probeProfile.mockResolvedValue({
      profile_id: "preset-1",
      profile_revision: 2,
      stages: [],
      capabilities: {
        endpoint_reachable: true,
        model_available: true,
        // A compatible provider may generate successfully without schema constraints.
        structured_output: false,
        embedding_available: true,
      },
    });
    activate.mockResolvedValue(undefined);
  });

  it("tests a preset only on the explicit test action and does not activate it", async () => {
    const user = userEvent.setup();
    render(<ModelSettings />);

    // Saving, testing and activation are separate user actions.
    await user.click(screen.getByRole("button", { name: "Test connection" }));

    expect(probeProfile).toHaveBeenCalledWith("preset-1");
    expect(activate).not.toHaveBeenCalled();
  });

  it("A failed probe neither activates nor fakes success", async () => {
    const user = userEvent.setup();
    // The real probeProfile never throws; it swallows failures into an error and returns undefined.
    // The first version of this test used mockRejectedValue, so it guarded behavior that does not exist in production.
    // Return a card whose probe failed instead; that is what really happens.
    probeProfile.mockResolvedValue({
      profile_id: "preset-1",
      profile_revision: 2,
      stages: [],
      capabilities: {
        endpoint_reachable: false,
        model_available: false,
        structured_output: false,
        embedding_available: false,
      },
    });
    render(<ModelSettings />);

    await user.click(screen.getByRole("button", { name: "Test connection" }));

    // Positive control: without this one, the previous assertion could be fooled by activating regardless of the probe result.
    expect(probeProfile).toHaveBeenCalledWith("preset-1");
    expect(activate).not.toHaveBeenCalled();
  });

  it("With insufficient balance you can still switch to this configuration, it just cannot activate generation", async () => {
    profiles = [
      {
        ...profiles[0],
        capabilities: {
          ...profile.capabilities,
          endpoint_reachable: true,
          model_available: true,
          embedding_available: true,
        },
      },
    ];
    const exhausted: ModelProbeResult = {
      profile_id: "preset-1",
      profile_revision: 2,
      stages: [
        {
          stage: "structured_output",
          ok: false,
          code: "insufficient_balance",
          latency_ms: 9,
        },
      ],
      capabilities: {
        endpoint_reachable: true,
        model_available: true,
        structured_output: false,
        embedding_available: true,
      },
    };
    probeProfile.mockResolvedValue(exhausted);
    probes = { "preset-1": exhausted };
    render(<ModelSettings />);
    const card = screen.getByRole("region", { name: preset.name });
    await userEvent.click(
      within(card).getByRole("button", { name: "Test connection" }),
    );
    expect(await screen.findByText(/balance is exhausted/)).toBeTruthy();
    await userEvent.click(
      within(card).getByRole("button", { name: "Set as current model" }),
    );
    expect(activate).toHaveBeenCalledWith("preset-1");
  });

  it("activates a tested preset only on the separate activation action", async () => {
    profiles = [
      {
        ...profiles[0],
        capabilities: {
          ...profile.capabilities,
          endpoint_reachable: true,
          model_available: true,
          embedding_available: true,
        },
      },
    ];
    render(<ModelSettings />);
    const card = screen.getByRole("region", { name: preset.name });
    await userEvent.click(
      within(card).getByRole("button", { name: "Set as current model" }),
    );
    expect(activate).toHaveBeenCalledWith("preset-1");
    expect(probeProfile).not.toHaveBeenCalled();
  });
});

it("keeps the form open and explains a 401 under the key input instead of collapsing it", async () => {
  const user = userEvent.setup();
  activate.mockResolvedValue(undefined);
  probes = {};
  profiles = [];
  create.mockResolvedValue({ ...profile, profile_id: "created", revision: 1 });
  probeProfile.mockResolvedValue({
    profile_id: "created",
    profile_revision: 1,
    stages: [
      { stage: "model", ok: false, code: "unauthorized", latency_ms: 3 },
    ],
    capabilities: {
      endpoint_reachable: true,
      model_available: false,
      structured_output: false,
      embedding_available: true,
    },
  });
  render(<ModelSettings />);
  await user.type(screen.getByLabelText("Configuration name"), "DeepSeek");
  await user.type(
    screen.getByLabelText("Generation model name"),
    "deepseek-flash",
  );
  await user.type(screen.getByLabelText(/^API Key/), "wrong-key");
  await user.click(
    screen.getByRole("button", { name: "Save, test and activate" }),
  );
  await act(async () => {});
  await act(async () => {});
  expect(await screen.findByLabelText(/^API Key/)).toBeTruthy();
  expect(
    screen.getByText(
      "The provider rejected this API Key (401). Make sure the key belongs to the service address you entered.",
    ),
  ).toBeTruthy();
  expect((screen.getByLabelText(/^API Key/) as HTMLInputElement).value).toBe(
    "wrong-key",
  );
  expect(screen.queryByText("Saved, connection passed")).toBeNull();
  expect(screen.queryByText("Saved, test failed")).toBeNull();
});
