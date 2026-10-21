// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { SourceDisconnectDialog } from "./SourceDisconnectDialog";

afterEach(cleanup);
it("requires explicit confirmation and supports safe keyboard cancellation", async () => {
  const cancel = vi.fn(),
    confirm = vi.fn();
  render(
    <SourceDisconnectDialog
      name="Cursor"
      onCancel={cancel}
      onConfirm={confirm}
    />,
  );
  expect(document.activeElement).toBe(
    screen.getByRole("button", { name: "Cancel" }),
  );
  await userEvent.keyboard("{Shift>}{Tab}{/Shift}");
  expect(document.activeElement).toBe(
    screen.getByRole("button", { name: "Revoke authorization" }),
  );
  await userEvent.keyboard("{Escape}");
  expect(cancel).toHaveBeenCalledOnce();
  expect(confirm).not.toHaveBeenCalled();
});
