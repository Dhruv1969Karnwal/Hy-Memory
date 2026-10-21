import { Component, type ReactNode } from "react";

type Props = { label: string; children: ReactNode };
type State = { crashed: boolean };

const RESET_MS = 15_000;

/**
 * 2026-10-12: when any home card fails to render, only that card falls, never the whole page.
 * Background: testers reported that one error also wrecked the layout. A failing card renders a quiet
 * placeholder card while everything else stays as it was; it remounts after 15 seconds and returns if the data recovered.
 * Usage: wrap every card in one and pass the card name used in the placeholder copy.
 */
export class HomeCardBoundary extends Component<Props, State> {
  state: State = { crashed: false };
  resetTimer: ReturnType<typeof setTimeout> | null = null;

  static getDerivedStateFromError(): State {
    return { crashed: true };
  }

  componentDidCatch(error: unknown) {
    console.error(`[home] ${this.props.label} render failed: `, error);
    if (this.resetTimer !== null) clearTimeout(this.resetTimer);
    this.resetTimer = setTimeout(() => {
      this.resetTimer = null;
      this.setState({ crashed: false });
    }, RESET_MS);
  }

  componentWillUnmount() {
    if (this.resetTimer !== null) clearTimeout(this.resetTimer);
  }

  render() {
    if (this.state.crashed) {
      return (
        <section className="r6-home-auto-failure" role="status">
          <p>
            {this.props.label} cannot be shown right now. Other features are
            unaffected and it recovers automatically shortly.
          </p>
        </section>
      );
    }
    return this.props.children;
  }
}
