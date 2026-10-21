// @vitest-environment jsdom
import type {
  SourceImportReceipt as Receipt,
  SourceResponse,
} from "@one-memory/contracts";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SourceImportReceipt } from "./SourceImportReceipt";

const getReceipt = vi.fn(),
  getConversation = vi.fn();
vi.mock("./api", () => ({
  getSourceImportReceipt: (...args: unknown[]) => getReceipt(...args),
  getConversation: (...args: unknown[]) => getConversation(...args),
}));
const source = {
  source_id: "cursor",
  display_name: "Cursor",
  authorized: true,
  imported_messages: 5,
} as SourceResponse;
const receipt: Receipt = {
  source_id: "cursor",
  measured_at: "2026-09-13T00:00:00Z",
  messages: 5,
  conversations: 2,
  person_messages: 2,
  agent_messages: 1,
  tool_messages: 1,
  unknown_author_messages: 1,
  unknown_time_messages: 4,
  examples: [
    {
      evidence_id: "e1",
      role: "user",
      written_by: "person",
      created_at: null,
      excerpt: "Budget <5, never >5.",
      truncated: true,
    },
  ],
};
beforeEach(() => {
  vi.resetAllMocks();
  getReceipt.mockResolvedValue(receipt);
  getConversation.mockResolvedValue({
    source_id: "cursor",
    conversation_id: "c1",
    messages: [
      {
        evidence_id: "e1",
        role: "user",
        written_by: "person",
        created_at: null,
        verbatim: "Full source text budget <5, never >5.",
      },
    ],
  });
});
afterEach(cleanup);

describe("source import receipt", () => {
  it("clears saved excerpts if opening the original conversation is denied", async () => {
    const user = userEvent.setup();
    render(<SourceImportReceipt source={source} />);
    await user.click(screen.getByText("View read content"));
    await screen.findByText("Budget <5, never >5.");
    getConversation.mockRejectedValue(new Error("revoked"));
    await user.click(
      screen.getByRole("button", { name: "View full conversation" }),
    );
    await screen.findByRole("alert");
    expect(screen.queryByText("Budget <5, never >5.")).toBeNull();
  });
  it("reads only after opening and shows measured authors and unknown message times", async () => {
    const user = userEvent.setup();
    render(<SourceImportReceipt source={source} />);
    expect(getReceipt).not.toHaveBeenCalled();
    await user.click(screen.getByText("View read content"));
    expect(await screen.findByText("Budget <5, never >5.")).toBeTruthy();
    expect(
      screen.getByText(
        /4 records have no original time, the import time was not substituted/,
      ),
    ).toBeTruthy();
    expect(
      screen.getByText("Original messages").nextElementSibling?.textContent,
    ).toBe("5");
    expect(
      screen.getByText("Unknown author").nextElementSibling?.textContent,
    ).toBe("1");
    expect(screen.getByText("User · Unknown time")).toBeTruthy();
    expect(screen.getByText(/only the first 600 characters/)).toBeTruthy();
  });
  it("opens full source conversation explicitly with authors preserved", async () => {
    const user = userEvent.setup();
    render(<SourceImportReceipt source={source} />);
    await user.click(screen.getByText("View read content"));
    await user.click(
      await screen.findByRole("button", { name: "View full conversation" }),
    );
    expect(
      await screen.findByText("Full source text budget <5, never >5."),
    ).toBeTruthy();
    expect(getConversation).toHaveBeenCalledWith("e1");
    await user.click(
      screen.getByRole("button", { name: "Close the full conversation" }),
    );
    expect(
      screen.queryByText("Full source text budget <5, never >5."),
    ).toBeNull();
  });
  it("drops a previous success while refresh is pending and on failure", async () => {
    const user = userEvent.setup();
    render(<SourceImportReceipt source={source} />);
    await user.click(screen.getByText("View read content"));
    await screen.findByText("Budget <5, never >5.");
    getReceipt.mockRejectedValue(new Error());
    await user.click(
      screen.getByRole("button", { name: "Refresh read content" }),
    );
    expect(screen.queryByText("Budget <5, never >5.")).toBeNull();
    expect(await screen.findByRole("alert")).toBeTruthy();
    getReceipt.mockResolvedValue(receipt);
    await user.click(
      screen.getByRole("button", { name: "Refresh read content" }),
    );
    expect(await screen.findByText("Budget <5, never >5.")).toBeTruthy();
  });
  it("revocation unmounts visible evidence and discards late responses", async () => {
    let resolve: (value: Receipt) => void = () => undefined;
    getReceipt.mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    const user = userEvent.setup();
    const view = render(<SourceImportReceipt source={source} />);
    await user.click(screen.getByText("View read content"));
    await waitFor(() => expect(getReceipt).toHaveBeenCalled());
    view.rerender(
      <SourceImportReceipt source={{ ...source, authorized: false }} />,
    );
    await act(async () => resolve(receipt));
    expect(screen.queryByText("View read content")).toBeNull();
    expect(screen.queryByText("Budget <5, never >5.")).toBeNull();
  });
  it("does not reopen a conversation after the disclosure is closed", async () => {
    let resolve: (value: unknown) => void = () => undefined;
    getConversation.mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    const user = userEvent.setup();
    render(<SourceImportReceipt source={source} />);
    await user.click(screen.getByText("View read content"));
    await user.click(
      await screen.findByRole("button", { name: "View full conversation" }),
    );
    await user.click(screen.getByText("View read content"));
    await act(async () =>
      resolve({
        source_id: "cursor",
        messages: [{ evidence_id: "e1", verbatim: "late original" }],
      }),
    );
    expect(screen.queryByText("late original")).toBeNull();
  });
  it("rejects source-mismatched responses", async () => {
    getReceipt.mockResolvedValue({ ...receipt, source_id: "another-source" });
    const user = userEvent.setup();
    render(<SourceImportReceipt source={source} />);
    await user.click(screen.getByText("View read content"));
    expect(await screen.findByRole("alert")).toBeTruthy();
    expect(screen.queryByText("Budget <5, never >5.")).toBeNull();
  });
});
