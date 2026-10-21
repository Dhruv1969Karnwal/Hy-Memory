// @vitest-environment jsdom
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import {
  OnboardingModelStep,
  type OnboardingModelStepProps,
} from "./OnboardingModelStep";

afterEach(cleanup);
function props(): OnboardingModelStepProps {
  return {
    draft: {
      provider_id: "fixture",
      name: "My profile",
      provider_kind: "openai_compatible",
      model: "user-model",
      base_url: "https://example.invalid/v1",
      api_key: "secret",
      disable_thinking: true,
      embedding_mode: "local",
      embedding_provider_kind: "openai_compatible",
      embedding_model: "",
      embedding_base_url: "",
      timeout_seconds: "240",
      max_concurrency: "4",
      context_window: "",
    },
    providers: [
      {
        id: "fixture",
        label: "User service",
        keyRequired: true,
        supported: true,
      },
    ],
    advancedOpen: false,
    onAdvancedChange: vi.fn(),
    onDraftChange: vi.fn(),
    phase: "idle",
    capability: { saveAndEnable: false },
    canTest: true,
    canSaveDraft: true,
    canSaveAndEnable: true,
    sourceNames: ["Source A"],
    importedConversationCount: 37,
    generationRecipient: "User service / generation",
    embeddingRecipient: "Local / embedding",
    onTest: vi.fn().mockResolvedValue(undefined),
    onSaveAndEnable: vi.fn().mockResolvedValue(undefined),
    onSaveDraft: vi.fn().mockResolvedValue(undefined),
    onBack: vi.fn(),
    onActionError: vi.fn(),
  };
}
it("renders step3 without engineering-only recipients, with unavailable enable failing closed", () => {
  const p = props();
  render(<OnboardingModelStep {...p} />);
  expect(
    screen.getByRole("heading", {
      name: "Pick the model and start distilling memories",
    }),
  ).toBeTruthy();
  expect(
    screen
      .getByRole("navigation", { name: "First-time setup progress" })
      .querySelector('[aria-current="step"]')?.textContent,
  ).toContain("Distill memories");
  fireEvent.click(screen.getByLabelText("Model provider"));
  expect(screen.getByRole("option", { name: "User service" })).toBeTruthy();
  expect(screen.queryByRole("option", { name: "DeepSeek" })).toBeNull();
  expect(screen.queryByText("User service / generation")).toBeNull();
  expect(screen.queryByText("Local / embedding")).toBeNull();
  expect(screen.getByText("Source A")).toBeTruthy();
  expect(
    screen.getAllByText(/A model cannot be enabled here right now/).length,
  ).toBeGreaterThan(0);
  fireEvent.click(screen.getByRole("button", { name: "Confirm and enable" }));
  expect(p.onSaveAndEnable).not.toHaveBeenCalled();
  expect(
    screen.queryByRole("checkbox", { name: /agree|allow sending/ }),
  ).toBeNull();
});
it("keeps a real multi-source summary compact", () => {
  const p = props();
  render(
    <OnboardingModelStep
      {...p}
      sourceNames={["WorkBuddy", "Cursor", "Codex", "Claude Code", "Pi"]}
    />,
  );
  expect(
    screen.getByText("WorkBuddy · Cursor · Codex and 5 apps in total"),
  ).toBeTruthy();
  expect(screen.queryByText("Pi")).toBeNull();
});
it("guides first-run keychain authorization when a saved key is unreadable", () => {
  const p = props();
  render(
    <OnboardingModelStep
      {...p}
      profile={{
        profile_id: "saved-profile",
        revision: 1,
        has_configured_api_key: true,
        credential_state: "unavailable",
      }}
    />,
  );
  expect(
    screen.getByText(/MemHub needs permission to access the keychain/),
  ).toBeTruthy();
  expect(screen.getByText(/Always allow/)).toBeTruthy();
});
it("draft edits and advanced mode are controlled; return never clears or saves values", () => {
  const p = props();
  const view = render(<OnboardingModelStep {...p} />);
  fireEvent.change(screen.getByLabelText("API Key"), {
    target: { value: "new-secret" },
  });
  expect(p.onDraftChange).toHaveBeenCalledWith({ api_key: "new-secret" });
  expect((screen.getByLabelText("API Key") as HTMLInputElement).value).toBe(
    "secret",
  );
  fireEvent.click(
    screen.getByRole("button", {
      name: "Configuration details and advanced settings",
    }),
  );
  expect(p.onAdvancedChange).toHaveBeenCalledWith(true);
  view.rerender(<OnboardingModelStep {...p} advancedOpen />);
  expect(screen.getByLabelText("API Key")).toHaveProperty("value", "secret");
  fireEvent.click(
    screen.getByRole("button", { name: "Back to provider selection" }),
  );
  expect(p.onAdvancedChange).toHaveBeenLastCalledWith(false);
  expect(
    screen.getByRole("heading", {
      name: "Configuration details, adjust as needed",
    }),
  ).toBeTruthy();
  expect(
    (screen.getByLabelText("Timeout (seconds)") as HTMLInputElement).max,
  ).toBe("600");
  expect(
    (screen.getByLabelText("Max concurrency") as HTMLInputElement).max,
  ).toBe("32");
  expect(
    (screen.getByLabelText("Context window") as HTMLInputElement).min,
  ).toBe("1024");
  fireEvent.click(
    screen.getByRole("button", { name: "Skip and go to the app" }),
  );
  expect(p.onBack).toHaveBeenCalledTimes(1);
  expect(p.onSaveDraft).not.toHaveBeenCalled();
  expect(p.draft.api_key).toBe("secret");
});
it("keeps skip available while an uncertain write is reconciled", () => {
  const p = props();
  render(
    <OnboardingModelStep
      {...p}
      advancedOpen
      inputsLocked
      capability={{ saveAndEnable: true }}
      enableActionLabel="Verify submission status"
    />,
  );
  expect(
    (screen.getByLabelText("Configuration name") as HTMLInputElement).disabled,
  ).toBe(true);
  expect(
    (
      screen.getByRole("button", {
        name: "Skip and go to the app",
      }) as HTMLButtonElement
    ).disabled,
  ).toBe(false);
  fireEvent.click(
    screen.getByRole("button", { name: "Skip and go to the app" }),
  );
  expect(p.onBack).toHaveBeenCalledOnce();
  expect(
    (
      screen.getByRole("button", {
        name: "Verify submission status",
      }) as HTMLButtonElement
    ).disabled,
  ).toBe(false);
});
it.each(["testing", "saving", "enabling"] as const)(
  "%s locks controlled actions and fields",
  (phase) => {
    const p = props();
    render(<OnboardingModelStep {...p} phase={phase} />);
    expect(
      (screen.getByLabelText("API Key") as HTMLInputElement).disabled,
    ).toBe(true);
    fireEvent.click(
      screen.getByRole("button", { name: "Skip and go to the app" }),
    );
    expect(p.onBack).not.toHaveBeenCalled();
    if (phase === "saving")
      expect(
        screen.queryByRole("status", { name: "Model submission status" }),
      ).toBeNull();
    else
      expect(
        screen.getByRole("status", { name: "Model submission status" }),
      ).toBeTruthy();
  },
);
it("one in-flight callback across double click, errors stay owner-controlled without exposing thrown secrets", async () => {
  const p = props();
  let reject!: (e: Error) => void;
  p.onSaveAndEnable = vi.fn(
    () =>
      new Promise<void>((_, r) => {
        reject = r;
      }),
  );
  render(
    <OnboardingModelStep
      {...p}
      capability={{ saveAndEnable: true }}
      probe={{ state: "passed", message: "Test passed" }}
    />,
  );
  const button = screen.getByRole("button", { name: "Confirm and enable" });
  fireEvent.click(button);
  fireEvent.click(button);
  expect(p.onSaveAndEnable).toHaveBeenCalledTimes(1);
  expect(screen.queryByRole("button", { name: "Test connection" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Save draft" })).toBeNull();
  await act(async () => reject(new Error("SECRET_TOKEN")));
  expect(p.onActionError).toHaveBeenCalledWith("save-and-enable");
  expect(screen.queryByText(/SECRET_TOKEN/)).toBeNull();
  await waitFor(() =>
    expect((button as HTMLButtonElement).disabled).toBe(false),
  );
});
it("renders field errors with links and preserves custom embedding in advanced mode", () => {
  const p = props();
  const view = render(
    <OnboardingModelStep
      {...p}
      error="The configuration changed; check it."
      fieldErrors={{ api_key: "The key failed verification" }}
    />,
  );
  const key = screen.getByLabelText("API Key");
  expect(key.getAttribute("aria-invalid")).toBe("true");
  expect(
    document.getElementById(key.getAttribute("aria-describedby") ?? "")
      ?.textContent,
  ).toContain("The key failed verification");
  view.rerender(
    <OnboardingModelStep
      {...p}
      advancedOpen
      draft={{
        ...p.draft,
        embedding_mode: "custom",
        embedding_model: "saved-vector",
        embedding_base_url: "http://127.0.0.1:11434",
      }}
    />,
  );
  expect(
    (screen.getByLabelText("Embedding model name") as HTMLInputElement).value,
  ).toBe("saved-vector");
  expect(
    screen.queryByRole("button", { name: "Collapse advanced configuration" }),
  ).toBeNull();
  expect(p.onDraftChange).not.toHaveBeenCalled();
});

it("discloses that testing does not send historical conversations", () => {
  render(<OnboardingModelStep {...props()} />);

  expect(
    screen.getByText(/Testing the connection only sends test content/),
  ).toBeTruthy();
  expect(screen.getByText(/Imported 37 conversations/)).toBeTruthy();
  expect(screen.getByText(/Once enabled, this model organizes/)).toBeTruthy();
});

it("does not expose an extra backlog selector in the first-run model page", () => {
  const p = props();
  render(<OnboardingModelStep {...p} capability={{ saveAndEnable: true }} />);
  fireEvent.click(screen.getByRole("button", { name: "Confirm and enable" }));
  expect(p.onSaveAndEnable).not.toHaveBeenCalled();
  expect(screen.getByText(/Test the model connection first/)).toBeTruthy();
  expect(screen.queryByLabelText("Backlog before enabling")).toBeNull();
});
it("separates connectivity testing from final enable", async () => {
  const p = props();
  render(<OnboardingModelStep {...p} capability={{ saveAndEnable: true }} />);
  expect(
    screen.getByRole("button", { name: "Test the model connection" }),
  ).toBeTruthy();
  expect(screen.queryByRole("button", { name: "Save draft" })).toBeNull();
  expect(
    screen.getByRole("button", { name: "Confirm and enable" }),
  ).toHaveProperty("disabled", true);
  fireEvent.click(
    screen.getByRole("button", { name: "Test the model connection" }),
  );
  await act(async () => {});
  expect(p.onTest).toHaveBeenCalledTimes(1);
  expect(p.onSaveDraft).not.toHaveBeenCalled();
  expect(p.onSaveAndEnable).not.toHaveBeenCalled();
});

it("requires owner validation as well as policy capability and truthful recipients", () => {
  const p = props();
  const view = render(
    <OnboardingModelStep
      {...p}
      capability={{ saveAndEnable: true }}
      canSaveAndEnable={false}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Confirm and enable" }));
  view.rerender(
    <OnboardingModelStep
      {...p}
      capability={{ saveAndEnable: true }}
      generationRecipient=""
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Confirm and enable" }));
  expect(p.onSaveAndEnable).not.toHaveBeenCalled();
});

it("opens the advanced design state with a route back to provider selection", () => {
  const p = props();
  const view = render(<OnboardingModelStep {...p} />);
  const toggle = screen.getByRole("button", {
    name: "Configuration details and advanced settings",
  });
  expect(toggle.getAttribute("aria-expanded")).toBe("false");
  expect(
    document.getElementById(toggle.getAttribute("aria-controls") ?? ""),
  ).toBeTruthy();
  toggle.focus();
  fireEvent.click(toggle);
  view.rerender(<OnboardingModelStep {...p} advancedOpen />);
  expect(
    screen.getByRole("button", { name: "Back to provider selection" }),
  ).toBeTruthy();
  expect(screen.getByLabelText("API Key")).toBeTruthy();
  expect(
    screen.getByText(
      "The presets are filled in; only custom services or special needs require changes.",
    ),
  ).toBeTruthy();
  expect(p.onDraftChange).not.toHaveBeenCalled();
});

it("uses the failure-state title and omits consent and scope-management UI", () => {
  const p = props();
  render(
    <OnboardingModelStep
      {...p}
      capability={{ saveAndEnable: true }}
      probe={{ state: "failed", message: "synthetic failure" }}
    />,
  );
  expect(
    screen.getByRole("heading", {
      name: "Connection failed; the configuration is kept",
    }),
  ).toBeTruthy();
  expect(
    screen.getByText(/The connection failed. Check the API Key/),
  ).toBeTruthy();
  expect(screen.getByText("synthetic failure")).toBeTruthy();
  expect(screen.queryByText("Adjust processing scope")).toBeNull();
  expect(
    screen.queryByText(/allow sending a small amount of test content/),
  ).toBeNull();
});

it("keeps the first-run model form free of a second consent strip", () => {
  const p = props();
  render(
    <OnboardingModelStep
      {...p}
      advancedOpen
      draft={{ ...p.draft, embedding_mode: "custom" }}
    />,
  );
  const fields = screen.getByRole("region", { name: "Model parameters" });
  expect(within(fields).getByLabelText("Embedding Base URL")).toBeTruthy();
  expect(
    screen.queryByText(/allow sending a small amount of test content/),
  ).toBeNull();
  expect(screen.queryByText(/auto-enable after it passes/)).toBeNull();
  expect(screen.queryByText(p.generationRecipient)).toBeNull();
  expect(screen.queryByText(p.embeddingRecipient)).toBeNull();
  expect(screen.queryByLabelText("Backlog before enabling")).toBeNull();
});
