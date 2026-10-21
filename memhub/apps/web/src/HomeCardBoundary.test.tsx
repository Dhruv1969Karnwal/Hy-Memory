// @vitest-environment jsdom

import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { HomeCardBoundary } from "./HomeCardBoundary";

function Boom(): never {
  throw new Error("boom");
}

it("a single card render error collapses only that card, sibling areas are unaffected", () => {
  const spy = vi.spyOn(console, "error").mockImplementation(() => {});
  render(
    <div>
      <HomeCardBoundary label="Memory overview">
        <Boom />
      </HomeCardBoundary>
      <p>normal area</p>
    </div>,
  );
  expect(
    screen.getByText(/Memory overview cannot be shown right now/),
  ).toBeTruthy();
  expect(screen.getByText("normal area")).toBeTruthy();
  spy.mockRestore();
});

it("a healthy card renders unchanged", () => {
  render(
    <HomeCardBoundary label="Runtime status">
      <p>Running in the background</p>
    </HomeCardBoundary>,
  );
  expect(screen.getByText("Running in the background")).toBeTruthy();
});
