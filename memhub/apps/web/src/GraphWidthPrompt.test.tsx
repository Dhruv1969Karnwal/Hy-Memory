// @vitest-environment jsdom

/** A-33: the width mismatch has to be a question, and the answer must default
 * to keeping what the user already has.
 *
 * The machine this ships to has a 768-dim library and a 512-dim built-in
 * model, so this prompt is not an edge case — it is the first screen after
 * install. Getting the default wrong here archives a live graph, which is the
 * accident this whole path exists to prevent.
 */

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GraphWidthPrompt } from "./GraphWidthPrompt";

const { getGraphWidth, decideGraphWidth } = vi.hoisted(() => ({
  getGraphWidth: vi.fn(),
  decideGraphWidth: vi.fn(),
}));

vi.mock("./api", () => ({
  apiErrorCode: () => null,
  getGraphWidth,
  decideGraphWidth,
}));

const MISMATCH = {
  mismatch: {
    stored_width: 768,
    expected_width: 512,
    graph_path: "/Users/me/.one-memory/hy-memory/graph.kuzu",
  },
  memory_count: 427,
};

describe("GraphWidthPrompt", () => {
  afterEach(cleanup);

  beforeEach(() => {
    vi.clearAllMocks();
    getGraphWidth.mockResolvedValue(MISMATCH);
    decideGraphWidth.mockResolvedValue({
      archived_path: null,
      exported_path: null,
      mismatch: null,
    });
  });

  it("moves focus into the modal and keeps Tab inside it", async () => {
    const user = userEvent.setup();
    render(
      <>
        <button type="button">Background action</button>
        <GraphWidthPrompt />
      </>,
    );
    const dialog = await screen.findByRole("dialog", {
      name: "The existing memory library does not match the built-in model",
    });
    const keep = screen.getByRole("radio", {
      name: /Keep the existing memory library/,
    });
    const confirm = screen.getByRole("button", { name: "Do that" });
    await waitFor(() => expect(document.activeElement).toBe(keep));

    await user.tab({ shift: false });
    expect(dialog.contains(document.activeElement)).toBe(true);
    await user.tab({ shift: false });
    expect(dialog.contains(document.activeElement)).toBe(true);
    await user.tab({ shift: false });
    expect(dialog.contains(document.activeElement)).toBe(true);
    expect(confirm).toBeTruthy();
  });

  describe("export first, then decide", () => {
    beforeEach(() => {
      decideGraphWidth.mockResolvedValue({
        archived_path: null,
        exported_path: "/Users/me/one-memory-graph-20260910T021341Z",
        // Exporting settles nothing, so the core still reports the mismatch.
        mismatch: MISMATCH.mismatch,
      });
    });

    async function exportFirst() {
      const user = userEvent.setup();
      render(<GraphWidthPrompt />);
      await user.click(
        await screen.findByRole("radio", {
          name: /Export first and decide later/,
        }),
      );
      await user.click(screen.getByRole("button", { name: "Do that" }));
      return user;
    }

    it("tells the user exactly where the backup actually landed", async () => {
      await exportFirst();

      expect(
        await screen.findByText(
          /\/Users\/me\/one-memory-graph-20260910T021341Z/,
        ),
      ).toBeTruthy();
    });

    it("stays on the three choices after exporting instead of stopping halfway", async () => {
      await exportFirst();

      await screen.findByText(/one-memory-graph-/);
      expect(
        screen.getByRole("dialog", {
          name: "The existing memory library does not match the built-in model",
        }),
      ).toBeTruthy();
      expect(
        screen.getByRole("radio", { name: /Switch to the built-in model/ }),
      ).toBeTruthy();
    });

    it("choosing rebuild after an export still goes through the one stated path", async () => {
      const user = await exportFirst();
      await screen.findByText(/one-memory-graph-/);
      decideGraphWidth.mockResolvedValue({
        archived_path: "/Users/me/graph.dim768.bak",
        exported_path: null,
        mismatch: null,
      });

      await user.click(
        screen.getByRole("radio", { name: /Switch to the built-in model/ }),
      );
      await user.click(screen.getByRole("button", { name: "Do that" }));

      await waitFor(() => {
        expect(decideGraphWidth).toHaveBeenLastCalledWith(
          "rebuild_with_new",
          expect.any(String),
        );
      });
      await waitFor(() => {
        expect(screen.queryByRole("dialog")).toBeNull();
      });
    });
  });

  it("keeping the existing library is what is selected before anyone clicks", async () => {
    render(<GraphWidthPrompt />);

    const keep = await screen.findByRole("radio", {
      name: /Keep the existing memory library/,
    });
    expect((keep as HTMLInputElement).checked).toBe(true);
    const rebuild = screen.getByRole("radio", {
      name: /Switch to the built-in model/,
    });
    expect((rebuild as HTMLInputElement).checked).toBe(false);
  });

  it("confirming without touching anything keeps the graph", async () => {
    const user = userEvent.setup();
    render(<GraphWidthPrompt />);

    await user.click(await screen.findByRole("button", { name: "Do that" }));

    await waitFor(() => {
      expect(decideGraphWidth).toHaveBeenCalledWith(
        "keep_existing",
        expect.any(String),
      );
    });
  });

  it("says what a rebuild costs, in memories rather than in dimensions", async () => {
    render(<GraphWidthPrompt />);

    const rebuild = await screen.findByRole("radio", {
      name: /Switch to the built-in model/,
    });
    expect(rebuild.getAttribute("aria-label") ?? "").toContain("427");
  });

  it("names both widths so the reader can tell which library is which", async () => {
    render(<GraphWidthPrompt />);

    const prompt = await screen.findByRole("dialog", {
      name: "The existing memory library does not match the built-in model",
    });
    expect(prompt.textContent).toContain("768");
    expect(prompt.textContent).toContain("512");
  });

  it("sends the rebuild only when the user picked it by name", async () => {
    const user = userEvent.setup();
    render(<GraphWidthPrompt />);

    await user.click(
      await screen.findByRole("radio", {
        name: /Switch to the built-in model/,
      }),
    );
    await user.click(screen.getByRole("button", { name: "Do that" }));

    await waitFor(() => {
      expect(decideGraphWidth).toHaveBeenCalledWith(
        "rebuild_with_new",
        expect.any(String),
      );
    });
  });

  it("stays out of the way when the widths agree", async () => {
    getGraphWidth.mockResolvedValue({ mismatch: null, memory_count: 427 });

    render(<GraphWidthPrompt />);

    await waitFor(() => {
      expect(getGraphWidth).toHaveBeenCalled();
    });
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("closes once the choice has been carried out", async () => {
    const user = userEvent.setup();
    render(<GraphWidthPrompt />);

    await user.click(await screen.findByRole("button", { name: "Do that" }));

    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });
  });
});
