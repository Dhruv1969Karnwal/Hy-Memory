import type {
  HomeSummary,
  LiveSyncHealth,
  MemoryProcessingSummary,
  RecentConversationItem,
  SourceResponse,
} from "@one-memory/contracts";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import {
  getHomeSummary,
  getLiveSyncHealth,
  getMemoryProcessing,
  getRecentConversations,
} from "./api";
import { AppMark } from "./app-mark";
import {
  type CallEvent,
  callDetailHeadline,
  callHostName,
  callRowDetail,
  getCallHistory,
} from "./call-history-client";
import { hostMatchesSource, sameConnectionHost } from "./connection-hosts";
import { readConnectionPreferences } from "./connection-preferences";
import {
  autoReconcile,
  hostApplyPending,
  retryAutoReconcile,
  useAutoConnect,
} from "./connection-reconcile";
import { HomeCardBoundary } from "./HomeCardBoundary";
import {
  hostBridgeAvailable,
  hostConnectionApproved,
  hostConnectionUnknown,
  useRecallHosts,
} from "./host-state";
import { subscribePoll } from "./poll-scheduler";
import { R6Button } from "./R6Controls";
import { prefetchDataRuntime } from "./runtime-state";
import type { useImportSources } from "./use-import-sources";
import "./applications-home.css";
import { countLabel } from "./count-label";

type HomeViewState = {
  expanded: boolean;
  homeScroll: number;
  allScroll: number;
  focus: string | null;
};

// The import controller outlives detail routes; a new App owner gets a fresh view.
// Keep only presentation state, never API data, DOM nodes or running animations.
/**
 * D96 item 4: the home "Recent calls" block stays as usual; only the entry into
 * the record page hides. `CallHistory.tsx` and the API are untouched. `onCalls`
 * must stay: the home page uses its presence to decide whether to fetch recent calls.
 */
export const CALL_HISTORY_ENTRY_VISIBLE = false;

// 2026-10-12: the Codex side once stored MCP debug frames like `[write_stdout] {"chars":...`
// as conversation summaries, so the home "Recent updates" exposed raw machine text. The
// presentation layer filters those frames out and prefers falling back to memories or the empty state over treating protocol frames as activity.
const DEBUG_FRAME_PREFIX = /^\s*\[(?:write_stdout|write_stderr|read_request)\b/;
function presentableDialogue(text: string): boolean {
  return !DEBUG_FRAME_PREFIX.test(text);
}

const retainedViews = new WeakMap<object, HomeViewState>();

function clockParts(value: string) {
  const ms = Date.parse(value);
  if (!Number.isFinite(ms)) return null;
  const date = new Date(ms);
  const now = new Date();
  return {
    clock: new Intl.DateTimeFormat("zh-CN", {
      hour: "2-digit",
      minute: "2-digit",
    }).format(date),
    today:
      date.getFullYear() === now.getFullYear() &&
      date.getMonth() === now.getMonth() &&
      date.getDate() === now.getDate(),
  };
}

function stamped(value: string, suffix = "") {
  const parts = clockParts(value);
  if (!parts) return `Time unknown${suffix}`;
  return `${parts.today ? "Today " : ""}${parts.clock}${suffix}`;
}

function memoryAction(action: string | undefined) {
  if (action === "create") return "Memory added";
  if (action === "correct" || action === "restore") return "Memory updated";
  return "Memory saved";
}

type ActivityChoice = "call" | "update";

function sourceMatchesHost(sourceId: string, hostId: string | null) {
  if (!hostId) return false;
  return hostMatchesSource(sourceId, hostId);
}

function successfulCall(item: CallEvent) {
  return item.state === "grounded" || item.state === "empty";
}

function callAttention(events: CallEvent[]) {
  const hosts = new Set<string>();
  for (const event of events) {
    const host = event.host_id;
    if (!host || hosts.has(host)) continue;
    hosts.add(host);
    if (event.state === "unavailable") {
      return { kind: "unavailable" as const, host };
    }
    let consecutiveTimeouts = 0;
    for (const candidate of events) {
      if (candidate.host_id !== host) continue;
      if (candidate.state !== "timed_out") break;
      consecutiveTimeouts += 1;
    }
    if (consecutiveTimeouts >= 3) {
      return { kind: "timeouts" as const, host };
    }
  }
  return null;
}

function sourceActivity(
  source: SourceResponse,
  events: CallEvent[],
  callHistoryUnavailable = false,
): string {
  if (!source.authorized) return "Not linked";
  const latest = events.find(
    (item) =>
      successfulCall(item) && sourceMatchesHost(source.source_id, item.host_id),
  );
  if (!latest) return "";
  const time = stamped(latest.occurred_at);
  const prefix = callHistoryUnavailable ? "Last read: " : "";
  return `${prefix}${callDetailHeadline(latest, source.display_name)} · ${time}`;
}

function sourceSyncActivity(sourceId: string, sync: LiveSyncHealth[]): string {
  const current = sync.find((item) => item.source_id === sourceId);
  return current?.last_success_at
    ? `Last sync ${stamped(current.last_success_at)}`
    : "";
}

function SourceMark({ source }: { source: SourceResponse }) {
  return (
    <AppMark
      id={source.source_id}
      label={source.display_name}
      fallbackClassName="r6-app-initial"
    />
  );
}

export function ApplicationsHome({
  imports,
  onManage,
  onMemories,
  onCalls,
  active = true,
}: {
  imports: ReturnType<typeof useImportSources>;
  onManage(sourceId?: string): void;
  onMemories(memoryId?: string): void;
  /** Kept for older callers; the "View conversation" entry was retired as required by product. */
  onConversations?(): void;
  onSettings?(): void;
  onCalls?(): void;
  active?: boolean;
}) {
  const hostQuery = useRecallHosts({ poll: active });
  const hostBridge = hostBridgeAvailable();
  const hosts = hostQuery.data ?? [];
  const hostState =
    hostQuery.data !== undefined && (hostBridge || hosts.length)
      ? "ready"
      : !hostBridge || hostQuery.isError
        ? "unavailable"
        : "loading";
  // A failed poll keeps the last answer on screen; the page must say it is old.
  const hostStale = hostQuery.isError && hostQuery.data !== undefined;
  const auto = useAutoConnect();
  const [summary, setSummary] = useState<HomeSummary | null>(null);
  const [processing, setProcessing] = useState<MemoryProcessingSummary | null>(
    null,
  );
  const [syncHealth, setSyncHealth] = useState<LiveSyncHealth[]>([]);
  const [failed, setFailed] = useState(false);
  const [recentCall, setRecentCall] = useState<CallEvent | null>(null);
  const [callEvents, setCallEvents] = useState<CallEvent[]>([]);
  const [callState, setCallState] = useState<
    "loading" | "ready" | "unavailable" | "disabled" | "incomplete"
  >("loading");
  const [callRevision, setCallRevision] = useState(0);
  const [latestDialogue, setLatestDialogue] =
    useState<RecentConversationItem | null>(null);
  const [dialogueState, setDialogueState] = useState<
    "loading" | "ready" | "unavailable"
  >("loading");
  const [activityView, setActivityView] = useState<ActivityChoice | null>(null);
  const [restartingBackground, setRestartingBackground] = useState(false);
  useEffect(() => {
    void callRevision;
    if (!active || !onCalls) return;
    let cancelled = false;
    let flight = false;
    const refresh = async () => {
      if (flight) return;
      flight = true;
      try {
        const result = await getCallHistory({ limit: 20, offset: 0 });
        if (cancelled) return;
        setCallEvents(result.items);
        setRecentCall(result.items.find(successfulCall) ?? null);
        setCallState(
          !result.available
            ? "disabled"
            : result.recording_error
              ? "incomplete"
              : "ready",
        );
      } catch {
        if (!cancelled) setCallState("unavailable");
      } finally {
        flight = false;
      }
    };
    void refresh();
    const stop = subscribePoll({
      key: "calls",
      intervalMs: 10_000,
      run: refresh,
    });
    return () => {
      cancelled = true;
      stop();
    };
  }, [active, onCalls, callRevision]);
  useEffect(() => {
    if (active) void prefetchDataRuntime();
  }, [active]);
  // Re-evaluated on every change; the apply itself only starts where the
  // preference and the real state differ and nothing was tried before.
  // biome-ignore lint/correctness/useExhaustiveDependencies: hostQuery.data is the trigger
  useEffect(() => {
    if (!active) return;
    void autoReconcile({
      sources: imports.sources,
      setSync: imports.setSync,
      refresh: imports.refresh,
    });
  }, [
    active,
    hostQuery.data,
    imports.refresh,
    imports.setSync,
    imports.sources,
  ]);
  const [retained] = useState(() => {
    const owner = imports.refresh ?? imports;
    let state = retainedViews.get(owner);
    if (!state) {
      state = { expanded: false, homeScroll: 0, allScroll: 0, focus: null };
      retainedViews.set(owner, state);
    }
    return state;
  });
  const [expanded, setExpanded] = useState(retained.expanded);
  const [phase, setPhase] = useState("idle");
  const [revision, setRevision] = useState(0);
  const expandButton = useRef<HTMLButtonElement>(null);
  const backButton = useRef<HTMLButtonElement>(null);
  const root = useRef<HTMLElement>(null);
  const overview = useRef<HTMLDivElement>(null);
  const list = useRef<HTMLElement>(null);
  const panel = useRef<HTMLElement>(null);
  const listId = useId();
  const transition = useRef<(next: boolean) => void>(() => {});
  const viewState = useRef(retained);
  const restorePending = useRef<(() => void) | null>(null);
  const touchStartY = useRef<number | null>(null);

  useLayoutEffect(() => {
    const home = root.current;
    const top = overview.current;
    const appsPanel = panel.current;
    const appList = list.current;
    if (!home || !top || !appsPanel || !appList) return;
    const saved = viewState.current;
    setExpanded(saved.expanded);
    setPhase("idle");
    if (!active) return;

    let scrollOwner = home.parentElement;
    while (
      scrollOwner &&
      !/(auto|scroll)/.test(getComputedStyle(scrollOwner).overflowY)
    ) {
      scrollOwner = scrollOwner.parentElement;
    }
    const scroller = scrollOwner ?? document.scrollingElement;
    let layoutExpanded = saved.expanded;
    let desired = saved.expanded;
    let generation = 0;
    let stopped = false;
    let animating = false;
    let animations: Animation[] = [];
    const media = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    const reduced = () =>
      media?.matches ||
      document.documentElement.dataset.nativeReduceMotion === "true";
    const cancel = () => {
      for (const animation of animations) animation.cancel();
      animations = [];
    };
    const animate = (
      element: HTMLElement,
      frames: Keyframe[],
      duration: number,
      easing = "cubic-bezier(.22,1,.36,1)",
    ) => {
      const animation = element.animate(frames, { duration, easing });
      animations.push(animation);
      return animation.finished.catch(() => {});
    };
    const rows = () =>
      Array.from(appList.querySelectorAll<HTMLElement>(".r6-home-row-content"));
    const restoreScroll = () => {
      if (scroller) scroller.scrollTop = desired ? 0 : saved.homeScroll;
      appList.scrollTop = desired ? saved.allScroll : 0;
    };
    const focusDestination = () => {
      const focused = document.activeElement;
      // Respect focus moved by the user to another control while motion was running.
      if (
        focused === document.body ||
        focused === expandButton.current ||
        focused === backButton.current ||
        focused === home
      ) {
        (desired ? backButton.current : expandButton.current)?.focus({
          preventScroll: true,
        });
      }
    };
    const commit = (next: boolean, nextPhase: string) => {
      layoutExpanded = next;
      flushSync(() => {
        setExpanded(next);
        setPhase(nextPhase);
      });
      restoreScroll();
    };
    const settle = (focus: boolean) => {
      ++generation;
      cancel();
      animating = false;
      commit(desired, "idle");
      if (focus) focusDestination();
    };
    const move = async (next: boolean) => {
      if (stopped || desired === next) return;
      const ticket = ++generation;
      const interrupted = animating;
      desired = next;
      saved.expanded = next;
      if (next && !layoutExpanded && !interrupted)
        saved.homeScroll = scroller?.scrollTop ?? 0;
      if (!next && layoutExpanded) saved.allScroll = appList.scrollTop;
      if (reduced() || !appsPanel.animate || document.hidden) {
        settle(true);
        return;
      }
      animating = true;
      if (next && !layoutExpanded && !interrupted) {
        cancel();
        flushSync(() => setPhase("overview-exit"));
        await animate(
          top,
          [
            { opacity: 1, transform: "translateY(0)" },
            { opacity: 0, transform: "translateY(-8px)" },
          ],
          90,
          "cubic-bezier(.4,0,.8,1)",
        );
        if (stopped || ticket !== generation) return;
      }
      // Read the current animated geometry before cancellation so reversal starts here.
      const before = appsPanel.getBoundingClientRect();
      const listBefore = appList.getBoundingClientRect();
      const rowBefore = rows().map((row) => row.getBoundingClientRect());
      cancel();
      commit(next, next ? "expanding" : "collapsing");
      const after = appsPanel.getBoundingClientRect();
      const delta = before.top - after.top;
      const rowAfter = rows().map((row) => row.getBoundingClientRect());
      const jobs = [
        animate(
          appsPanel,
          [
            {
              transform: `translateY(${delta}px)`,
              height: `${before.height}px`,
            },
            { transform: "translateY(0)", height: `${after.height}px` },
          ],
          190,
        ),
      ];
      rows().forEach((row, index) => {
        const previous = rowBefore[index];
        const current = rowAfter[index];
        const wasVisible =
          previous?.height &&
          previous.bottom > listBefore.top &&
          previous.top < listBefore.bottom;
        const shift = wasVisible ? previous.top - current.top - delta : 8;
        jobs.push(
          animate(
            row,
            [
              {
                transform: `translateY(${shift}px)`,
                opacity: wasVisible ? 1 : 0,
              },
              { transform: "translateY(0)", opacity: 1 },
            ],
            190,
          ),
        );
      });
      await Promise.all(jobs);
      if (stopped || ticket !== generation) return;
      cancel();
      if (!next) {
        flushSync(() => setPhase("overview-enter"));
        await animate(
          top,
          [
            { opacity: 0, transform: "translateY(-8px)" },
            { opacity: 1, transform: "translateY(0)" },
          ],
          100,
        );
        if (stopped || ticket !== generation) return;
      }
      cancel();
      animating = false;
      setPhase("idle");
      focusDestination();
    };
    transition.current = (next) => {
      void move(next);
    };
    const preferenceChanged = () => {
      if (animating && reduced()) settle(true);
    };
    const resized = () => {
      if (animating) settle(false);
    };
    const visibilityChanged = () => {
      if (document.hidden && animating) settle(false);
    };
    const keyDown = (event: KeyboardEvent) => {
      if (
        event.key === "Escape" &&
        desired &&
        !event.defaultPrevented &&
        event.target instanceof Node &&
        (home.contains(event.target) ||
          (animating && event.target === document.body))
      ) {
        event.preventDefault();
        void move(false);
      }
    };
    const rememberHomeScroll = () => {
      if (!desired && !animating && scroller)
        saved.homeScroll = scroller.scrollTop;
    };
    const observer = new MutationObserver(preferenceChanged);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-native-reduce-motion"],
    });
    media?.addEventListener("change", preferenceChanged);
    window.addEventListener("resize", resized);
    document.addEventListener("visibilitychange", visibilityChanged);
    document.addEventListener("keydown", keyDown);
    scroller?.addEventListener("scroll", rememberHomeScroll);
    restorePending.current = () => {
      // Activity can hide during the exit stage, before expanded state is committed.
      if (home.dataset.expanded !== String(saved.expanded)) return;
      restorePending.current = null;
      restoreScroll();
      if (!saved.focus) return;
      const destination = Array.from(
        home.querySelectorAll<HTMLElement>("[data-home-focus]"),
      ).find(
        (element) =>
          element.dataset.homeFocus === saved.focus &&
          !element.closest("[hidden]"),
      );
      (
        destination ?? (desired ? backButton.current : expandButton.current)
      )?.focus({ preventScroll: true });
    };
    return () => {
      stopped = true;
      ++generation;
      cancel();
      transition.current = () => {};
      restorePending.current = null;
      observer.disconnect();
      media?.removeEventListener("change", preferenceChanged);
      window.removeEventListener("resize", resized);
      document.removeEventListener("visibilitychange", visibilityChanged);
      document.removeEventListener("keydown", keyDown);
      scroller?.removeEventListener("scroll", rememberHomeScroll);
    };
  }, [active]);
  useLayoutEffect(() => {
    restorePending.current?.();
  });
  const stale = imports.sourceSnapshot?.stale === true;
  const sourceVersion = imports.sources
    .map((s) => `${s.source_id}:${s.authorized}:${s.grant_revision}`)
    .join("|");
  const requestKey = `${sourceVersion}:${revision}`;
  // requestKey retriggers this read when the source snapshot or a manual retry changes.
  // biome-ignore lint/correctness/useExhaustiveDependencies: requestKey is the refresh token
  useEffect(() => {
    if (!active) return;
    let stopped = false,
      busy = false;
    async function refresh() {
      if (busy || document.hidden) return;
      busy = true;
      const [home, state, sync, recent] = await Promise.allSettled([
        getHomeSummary(),
        getMemoryProcessing(),
        getLiveSyncHealth(),
        getRecentConversations({ limit: 1, recentActivity: true }),
      ]);
      busy = false;
      if (stopped) return;
      if (home.status === "fulfilled") {
        setFailed(false);
        // stale=true means this read timed out and its numbers do not describe the present:
        // keep the previous overview, let the next poll 10 seconds later fill it in, never reset to zero or report an error.
        if (!home.value.stale) setSummary(home.value);
      } else {
        setFailed(true);
      }
      if (state.status === "fulfilled") setProcessing(state.value);
      if (sync.status === "fulfilled") setSyncHealth(sync.value);
      if (recent.status === "fulfilled") {
        setLatestDialogue(recent.value.items[0] ?? null);
        setDialogueState("ready");
      } else {
        setDialogueState("unavailable");
      }
    }
    void refresh();
    const stop = subscribePoll({
      key: "applications",
      intervalMs: 10_000,
      run: refresh,
    });
    return () => {
      stopped = true;
      stop();
    };
  }, [active, requestKey]);
  const visible = summary;
  const latest = visible?.latest;
  const summaryUnavailable = failed;
  const hasCall =
    recentCall !== null &&
    (callState === "ready" || callState === "incomplete");
  const hasMemory = latest !== null && latest !== undefined;
  const dialogueReady =
    dialogueState === "ready" &&
    latestDialogue !== null &&
    presentableDialogue(latestDialogue.excerpt);
  const updateKind: "memory" | "dialogue" | null = dialogueReady
    ? "dialogue"
    : hasMemory
      ? "memory"
      : null;
  const availableActivities: ActivityChoice[] = [
    ...(hasCall ? (["call"] as const) : []),
    ...(updateKind ? (["update"] as const) : []),
  ];
  const pinned =
    activityView !== null && availableActivities.includes(activityView)
      ? activityView
      : null;
  const historyImporting = imports.jobs.some(
    (job) => job.status === "running" || job.status === "pending",
  );
  const recentCallSourceId = (recentCall?.references ?? []).find(
    (reference) => reference.source_id,
  )?.source_id;
  const recentCallSource = imports.sources.find(
    (source) => source.source_id === recentCallSourceId,
  );
  const recentCallHost = imports.sources.find((source) =>
    sourceMatchesHost(source.source_id, recentCall?.host_id ?? null),
  );
  const attention = callAttention(callEvents);
  const attentionHost = attention ? callHostName(attention.host) : null;
  const activity = summaryUnavailable
    ? "unavailable"
    : visible === null
      ? "loading"
      : pinned === "call"
        ? "call"
        : pinned === "update"
          ? updateKind
          : updateKind
            ? updateKind
            : hasCall
              ? "call"
              : callState === "unavailable"
                ? "unavailable"
                : historyImporting
                  ? "importing"
                  : callState === "loading"
                    ? recentCall !== null
                      ? "call"
                      : "loading"
                    : processing?.running
                      ? "first-running"
                      : processing?.pending || processing?.waiting_for_model
                        ? "first-queued"
                        : "empty";
  const apps = imports.sources.filter(
    (s) => s.authorized || s.detected === true || s.location_state === "found",
  );
  const stateLabel = processing?.running
    ? "Organizing"
    : processing?.blocked_stages?.some((stage) => stage.count > 0)
      ? "Queued for processing"
      : processing?.waiting_for_model
        ? "Waiting for the model"
        : processing?.failed
          ? "Needs attention"
          : processing?.pending
            ? "Preparing"
            : processing
              ? "All set"
              : "Pending verification";
  const latestClock = latest ? clockParts(latest.updated_at) : null;
  const time = latestClock?.clock ?? (latest ? "Time unknown" : null);
  const checked = visible ? clockParts(visible.observed_at) : null;
  // A failed read leaves the last health in place on purpose: clearing it would
  // drop the instance it names, and that instance is what keeps memory writes
  // and backfill receipts attached to the run that produced them. So the card
  // needs its own answer to "is this still true", rather than reading one off
  // an object that is being kept precisely because it may be out of date.
  const connected = Boolean(imports.health) && !imports.healthStale;
  const admitted = hosts.filter(hostConnectionApproved).length;
  const unchecked = hosts.filter(hostConnectionUnknown);
  const hostReadFailed = hostBridge && hostState === "unavailable";
  const preferences = readConnectionPreferences();
  const accessPending =
    hostState === "loading" ||
    (!hostBridge && hostState !== "ready") ||
    (admitted === 0 && hostApplyPending(auto, preferences.hostSourceIds));
  const accessText = accessPending
    ? "Verifying"
    : hostReadFailed
      ? "Pending verification"
      : !unchecked.length
        ? `${countLabel(admitted, "app")} connected`
        : admitted
          ? `${countLabel(admitted, "app")} connected, ${unchecked.length} pending verification`
          : `${countLabel(unchecked.length, "app")} pending verification`;
  const accessUnknownText = accessPending
    ? null
    : hostReadFailed
      ? "App connection status could not be read right now, so the number of connected apps cannot be confirmed."
      : unchecked.length
        ? `The memory permission for ${unchecked.map((host) => host.label).join(", ")} could not be read right now, so the connection cannot be confirmed.`
        : null;
  async function restartBackground() {
    if (restartingBackground) return;
    setRestartingBackground(true);
    try {
      if (!window.oneMemory?.restartBackground)
        throw new Error("runtime_restart_unavailable");
      await window.oneMemory.restartBackground();
      setRevision((value) => value + 1);
      setCallRevision((value) => value + 1);
    } catch {
      setCallRevision((value) => value + 1);
    } finally {
      setRestartingBackground(false);
    }
  }
  const nameOf = (sourceId: string) =>
    imports.sources.find((source) =>
      sameConnectionHost(source.source_id, sourceId),
    )?.display_name ?? sourceId;
  const autoFailures = auto.failures.filter((failure) =>
    failure.kind === "host"
      ? preferences.hostSourceIds.some((id) =>
          sameConnectionHost(id, failure.sourceId),
        ) &&
        !hosts.some(
          (host) =>
            hostConnectionApproved(host) &&
            hostMatchesSource(failure.sourceId, host.hostId),
        )
      : preferences.syncSourceIds.includes(failure.sourceId) &&
        !imports.sources.some(
          (source) =>
            source.source_id === failure.sourceId && source.sync_enabled,
        ),
  );
  const autoFailureText = autoFailures
    .map((failure) =>
      failure.kind === "host"
        ? `${nameOf(failure.sourceId)} could not be connected automatically${failure.reason ? `: ${failure.reason}` : ""}.`
        : `Continuous sync for ${nameOf(failure.sourceId)} did not turn on automatically. ${failure.reason ?? ""}`,
    )
    .join("");
  const authorizedSyncSources = imports.sources.filter(
    (source) => source.authorized && source.sync_enabled,
  );
  const wantsSync = preferences.syncSourceIds.length > 0;
  const syncApplyFailed = autoFailures.some(
    (failure) => failure.kind === "sync",
  );
  const syncState = authorizedSyncSources.length
    ? `${countLabel(authorizedSyncSources.length, "app")} enabled`
    : syncApplyFailed
      ? "Not enabled"
      : wantsSync || !visible
        ? "Checking status"
        : "Not enabled yet";
  function toggleAll(next: boolean) {
    transition.current(next);
  }
  return (
    <section
      className="r6-home"
      ref={root}
      aria-label="App home"
      data-expanded={expanded}
      data-phase={phase}
      onFocusCapture={(event) => {
        viewState.current.focus =
          (event.target as HTMLElement).dataset.homeFocus ?? null;
      }}
    >
      <header className="r6-home-heading">
        <h1>{expanded ? "All apps" : "Apps"}</h1>
      </header>
      <button
        type="button"
        ref={backButton}
        data-home-focus="back"
        className="r6-home-back r6-button"
        hidden={!expanded}
        data-variant="quiet"
        onClick={() => toggleAll(false)}
      >
        <span className="r6-button-label">
          <ArrowLeft size={16} aria-hidden="true" />
          Back to home
        </span>
      </button>
      <div
        className="r6-home-top"
        ref={overview}
        hidden={expanded}
        inert={expanded || phase !== "idle"}
      >
        <HomeCardBoundary label="Memory overview">
          <section className="r6-home-overview" aria-label="Memory overview">
            <header>
              <h2>Memory overview</h2>
            </header>
            <div className="r6-home-metrics">
              <div>
                <p>Memories distilled</p>
                {/* The figure slot used to hold the word "Loading" while the first
                  read was in flight, which is how "the big number is missing" (D51) shows up
                  on a slow link: the biggest type on the page spends the first
                  frame saying something that is not a number. A bar of the same
                  height reserves the digits instead, and the wording moves to a
                  live region so screen readers still hear it. "Temporarily unreadable" stays a
                  word, because that is a settled state rather than progress. */}
                {!visible && !summaryUnavailable ? (
                  <>
                    <p className="r6-home-figure">
                      <span
                        className="r6-home-figure-pending"
                        aria-hidden="true"
                      />
                    </p>
                    <small role="status">
                      Loading the number of distilled memories...
                    </small>
                  </>
                ) : (
                  <>
                    <p className="r6-home-figure">
                      <strong className={visible ? "" : "r6-home-word"}>
                        {visible
                          ? new Intl.NumberFormat("zh-CN").format(
                              visible.memory_count,
                            )
                          : "Temporarily unreadable"}
                      </strong>
                      {visible ? <i>memories</i> : null}
                    </p>
                    <small>Used within each app's authorized scope</small>
                  </>
                )}
              </div>
              <div>
                <p>Distillation status</p>
                <strong className="r6-home-word">{stateLabel}</strong>
                <small>
                  {processing?.running
                    ? "New conversations are being organized"
                    : processing?.pending
                      ? "Waiting for the queue to start"
                      : "Based on the current processing status"}
                </small>
              </div>
              <div>
                <p>Recent updates</p>
                <strong className={time ? "r6-home-time" : "r6-home-word"}>
                  {time ??
                    (visible ? "Not generated yet" : "Pending verification")}
                </strong>
                <small>
                  {time
                    ? `${latestClock?.today ? "Today - " : ""}Memory saved`
                    : "History and memories are handled separately"}
                </small>
              </div>
            </div>
            <div className="r6-home-activity">
              {availableActivities.length > 1 &&
              (activity === "call" ||
                activity === "memory" ||
                activity === "dialogue") ? (
                <div className="r6-home-activity-tabs" role="tablist">
                  {(["update", "call"] as const).map((item) => (
                    <button
                      type="button"
                      role="tab"
                      key={item}
                      aria-selected={
                        item === "call"
                          ? activity === "call"
                          : activity === "memory" || activity === "dialogue"
                      }
                      onClick={() => setActivityView(item)}
                    >
                      {item === "call" ? "Recent calls" : "Recent updates"}
                    </button>
                  ))}
                </div>
              ) : (
                <p>
                  {activity === "unavailable"
                    ? "Activity is unavailable"
                    : activity === "call"
                      ? "Recent calls"
                      : activity === "dialogue" || activity === "memory"
                        ? "Recent updates"
                        : "First distillation"}
                </p>
              )}
              <div>
                {activity === "call" && recentCall ? (
                  <div className="r6-home-call-direction" aria-hidden="true">
                    {recentCallSource ? (
                      <>
                        <AppMark id={recentCallSource.source_id} size={26} />
                        <ArrowRight size={22} />
                      </>
                    ) : null}
                    <AppMark
                      id={
                        recentCallHost?.source_id ??
                        recentCall.host_id ??
                        "unknown"
                      }
                      size={26}
                    />
                  </div>
                ) : activity === "memory" && latest?.source_ids[0] ? (
                  (() => {
                    const source = imports.sources.find(
                      (item) => item.source_id === latest.source_ids[0],
                    );
                    return source ? <SourceMark source={source} /> : null;
                  })()
                ) : activity === "dialogue" && latestDialogue ? (
                  (() => {
                    const source = imports.sources.find(
                      (item) => item.source_id === latestDialogue.source_id,
                    );
                    return source ? <SourceMark source={source} /> : null;
                  })()
                ) : null}
                <section>
                  <h3>
                    {activity === "unavailable"
                      ? callState === "unavailable"
                        ? "Call records cannot be read right now"
                        : "Memory status cannot be read right now"
                      : activity === "call" && recentCall
                        ? callDetailHeadline(
                            recentCall,
                            recentCallHost?.display_name ??
                              callHostName(recentCall.host_id),
                          )
                        : activity === "memory" && latest
                          ? `${latest.content}${latest.truncated ? "…" : ""}`
                          : activity === "dialogue" && latestDialogue
                            ? `${latestDialogue.excerpt}${latestDialogue.truncated ? "…" : ""}`
                            : activity === "importing"
                              ? "Conversations are still importing; your memories will be distilled shortly"
                              : activity === "first-running"
                                ? "Distilling the first memories"
                                : activity === "first-queued"
                                  ? "Preparing to distill the first memories"
                                  : activity === "loading"
                                    ? "Loading activity"
                                    : "No memories generated yet"}
                  </h3>
                  <small>
                    {activity === "unavailable"
                      ? callState === "unavailable"
                        ? "Existing memories remain available; retrying does not start a new call."
                        : "A failed read is never treated as having no memories."
                      : activity === "call" && recentCall
                        ? `${stamped(recentCall.occurred_at)} · ${callRowDetail(recentCall)}${callState === "incomplete" ? " · a few calls were not recorded, so the list may be incomplete" : ""}`
                        : activity === "memory" && latest
                          ? `${time} - ${memoryAction(latest.action)} - ${latest.source_ids.map((id) => imports.sources.find((s) => s.source_id === id)?.display_name ?? id).join(", ")} - version ${latest.revision}`
                          : activity === "dialogue" && latestDialogue
                            ? `${clockParts(latestDialogue.created_at ?? latestDialogue.imported_at)?.clock ?? "Time unknown"} ${latestDialogue.time_basis === "conversation" ? "Conversation time" : "Import time"} - ${imports.sources.find((source) => source.source_id === latestDialogue.source_id)?.display_name ?? latestDialogue.source_id}`
                            : activity === "importing"
                              ? "Conversations are still being saved in the background."
                              : "History is saved; generated memories will appear here."}
                  </small>
                </section>
                {(activity === "unavailable" ||
                  (activity === "call" && CALL_HISTORY_ENTRY_VISIBLE) ||
                  activity === "memory") && (
                  <button
                    type="button"
                    className="r6-home-activity-link"
                    onClick={
                      activity === "unavailable"
                        ? () => {
                            setRevision((n) => n + 1);
                            setCallRevision((n) => n + 1);
                          }
                        : activity === "call"
                          ? onCalls
                          : () => onMemories(latest?.memory_id)
                    }
                  >
                    {activity === "unavailable"
                      ? "Reload"
                      : activity === "call"
                        ? "View records"
                        : "View memories"}
                    {activity === "unavailable" ? null : (
                      <ArrowRight size={14} aria-hidden="true" />
                    )}
                  </button>
                )}
              </div>
            </div>
          </section>
        </HomeCardBoundary>
        {attention ? (
          <section className="r6-home-auto-failure" role="alert">
            <p>
              {attention.kind === "unavailable"
                ? "The background service is not running, so apps cannot query memories right now"
                : `${attentionHost} found no memories in its last few calls; the background service may be busy and will recover on its own`}
            </p>
            {attention.kind === "unavailable" || CALL_HISTORY_ENTRY_VISIBLE ? (
              <R6Button
                variant="quiet"
                pending={
                  attention.kind === "unavailable" && restartingBackground
                }
                onClick={
                  attention.kind === "unavailable"
                    ? () => void restartBackground()
                    : onCalls
                }
              >
                {attention.kind === "unavailable" ? "Restart" : "View records"}
              </R6Button>
            ) : null}
          </section>
        ) : null}
        <HomeCardBoundary label="Runtime status">
          <aside className="r6-home-runtime">
            <h2>Runtime status</h2>
            <h3>
              {connected
                ? "Backend running"
                : imports.healthStale
                  ? "Connection status pending verification"
                  : "Service not connected"}
            </h3>
            <p>Only authorized scopes are processed</p>
            <dl>
              <div>
                <dt>Conversation sync</dt>
                <dd>{syncState}</dd>
              </div>
              <div>
                <dt>Memory organization</dt>
                <dd>{stateLabel}</dd>
              </div>
              <div>
                <dt>App access</dt>
                <dd>{accessText}</dd>
              </div>
            </dl>
            {accessUnknownText ? (
              <div
                role="status"
                aria-label="App connection verification"
                className="r6-home-auto-failure"
              >
                <p>{accessUnknownText}</p>
                <button
                  type="button"
                  className="r6-home-activity-link"
                  disabled={hostQuery.isFetching}
                  onClick={() => void hostQuery.refetch()}
                >
                  {hostQuery.isFetching ? "Verifying..." : "Verify again"}
                </button>
              </div>
            ) : null}
            {hostStale ? (
              <p
                role="status"
                aria-label="App connection status"
                className="r6-home-auto-failure"
              >
                App connection status could not be read right now; the last
                known state is shown.
              </p>
            ) : null}
            {autoFailureText ? (
              <div role="alert" className="r6-home-auto-failure">
                <p>{`${autoFailureText} It will not retry automatically.`}</p>
                <button
                  type="button"
                  className="r6-home-activity-link"
                  disabled={auto.running.length > 0}
                  onClick={() =>
                    void retryAutoReconcile({
                      sources: imports.sources,
                      setSync: imports.setSync,
                      refresh: imports.refresh,
                    })
                  }
                >
                  Reconnect
                </button>
              </div>
            ) : null}
            <p className="r6-home-checked">
              {!checked
                ? "Last check: pending verification"
                : `Last check: ${checked.today ? "Today " : ""}${checked.clock}`}
            </p>
          </aside>
        </HomeCardBoundary>
      </div>
      <section
        className="r6-home-apps"
        ref={panel}
        aria-label="My apps"
        onWheel={(event) => {
          if (!expanded && apps.length > 3 && event.deltaY > 10)
            toggleAll(true);
          else if (
            expanded &&
            event.deltaY < -10 &&
            (list.current?.scrollTop ?? 0) <= 0
          )
            toggleAll(false);
        }}
        onTouchStart={(event) => {
          touchStartY.current = event.touches[0]?.clientY ?? null;
        }}
        onTouchEnd={(event) => {
          const start = touchStartY.current;
          touchStartY.current = null;
          if (start === null) return;
          const end = event.changedTouches[0]?.clientY;
          if (end === undefined) return;
          const distance = start - end;
          if (!expanded && apps.length > 3 && distance > 24) toggleAll(true);
          else if (
            expanded &&
            distance < -24 &&
            (list.current?.scrollTop ?? 0) <= 0
          )
            toggleAll(false);
        }}
      >
        <header>
          <h2>{expanded ? `${countLabel(apps.length, "app")}` : "My apps"}</h2>
          {apps.length > 3 ? (
            <button
              type="button"
              ref={expandButton}
              data-home-focus="all"
              hidden={expanded}
              aria-expanded={expanded}
              aria-controls={listId}
              className="r6-home-expand"
              onClick={() => toggleAll(true)}
            >
              View all {countLabel(apps.length, "app")}
            </button>
          ) : null}
        </header>
        <section
          className="r6-home-list"
          ref={list}
          id={listId}
          aria-label="App list"
          tabIndex={expanded ? 0 : -1}
          onScroll={(event) => {
            if (expanded)
              viewState.current.allScroll = event.currentTarget.scrollTop;
          }}
        >
          {(expanded ? apps : apps.slice(0, 3)).map((source) => {
            const activity =
              sourceActivity(source, callEvents, callState === "unavailable") ||
              sourceSyncActivity(source.source_id, syncHealth);
            const host = hosts.find((item) =>
              sourceMatchesHost(source.source_id, item.hostId),
            );
            return (
              <div className="r6-home-app-row" key={source.source_id}>
                <div className="r6-home-row-content">
                  <SourceMark source={source} />
                  <section>
                    <h3>{source.display_name}</h3>
                    <p>
                      {source.sync_enabled
                        ? "Continuous sync is on"
                        : host?.installed &&
                            host.permission?.state === "approved"
                          ? "May request authorized memories"
                          : source.authorized
                            ? "History reading is authorized"
                            : "Connection incomplete"}
                    </p>
                  </section>
                  {activity ? (
                    <p className="r6-home-app-meta">{activity}</p>
                  ) : null}
                  <R6Button
                    variant="quiet"
                    data-home-focus={`source:${source.source_id}`}
                    onClick={() => onManage(source.source_id)}
                  >
                    View details
                  </R6Button>
                </div>
              </div>
            );
          })}
          {!apps.length ? (
            <p className="r6-home-empty">
              {stale
                ? "The app list is unreadable right now"
                : "No apps to show yet"}
            </p>
          ) : null}
        </section>
      </section>
    </section>
  );
}
