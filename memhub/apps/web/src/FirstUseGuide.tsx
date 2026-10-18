import type {
  GovernedMemory,
  MemoryDetail,
  SourceResponse,
} from "@one-memory/contracts";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  Copy,
  ExternalLink,
  FileText,
  Link2,
  RefreshCw,
  X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { getMemories, getMemory } from "./api";
import { copyText } from "./copy-text";
import {
  draftForScenario,
  GUIDE_SCENARIOS,
  type GuideRoute,
  type GuideScenario,
  guideDetailAvailable,
  guideHosts,
  sameGuideMemory,
} from "./first-use-model";
import type { RecallHostStatus } from "./global";
import { refreshRecallHosts, useRecallHosts } from "./host-state";
import { refreshSources } from "./source-state";

const NO_HOSTS: RecallHostStatus[] = [];

export function FirstUseGuide({
  sources,
  onSources,
  onClose,
}: {
  sources: SourceResponse[];
  onSources(): void;
  onClose(): void;
}) {
  const [scene, setScene] = useState<GuideScenario>("same");
  const [route, setRoute] = useState<GuideRoute>("context");
  const [memories, setMemories] = useState<GovernedMemory[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState<MemoryDetail | null>(null);
  const hostRead = useRecallHosts();
  const refetchHosts = hostRead.refetch;
  const hostsError = hostRead.isError;
  const hosts = hostsError ? NO_HOSTS : (hostRead.data ?? NO_HOSTS);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");
  const [reload, setReload] = useState(0);
  const [targetId, setTargetId] = useState("");
  const [topic, setTopic] = useState("");
  const [notice, setNotice] = useState("");
  const [copiedKey, setCopiedKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [reported, setReported] = useState<"ok" | "missing" | null>(null);
  const mounted = useRef(true);
  const actionEpoch = useRef(0);
  const contextKey = JSON.stringify([
    scene,
    route,
    selectedId,
    detail?.memory.revision,
    targetId,
    topic,
    reload,
    sources.map((item) => [item.source_id, item.authorized]),
  ]);
  const contextRef = useRef(contextKey);
  contextRef.current = contextKey;
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      actionEpoch.current++;
    };
  }, []);

  // biome-ignore lint/correctness/useExhaustiveDependencies: reload is the explicit user refresh token.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    void refetchHosts();
    getMemories()
      .then((items) => {
        if (cancelled) return;
        const usable = items.filter(
          (item) =>
            item.status === "confirmed" &&
            !item.deleted_at &&
            item.evidence_ids.length > 0,
        );
        setMemories(usable);
        setSelectedId((id) =>
          usable.some((item) => item.memory_id === id)
            ? id
            : (usable[0]?.memory_id ?? ""),
        );
      })
      .catch(() => {
        if (!cancelled) {
          setMemories([]);
          setSelectedId("");
          setError("Memories cannot be read right now. Try again later.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [reload]);

  useEffect(() => {
    let cancelled = false;
    setDetail(null);
    setError("");
    setNotice("");
    setCopiedKey("");
    setReported(null);
    if (!selectedId) {
      setDetailLoading(false);
      return;
    }
    setDetailLoading(true);
    getMemory(selectedId)
      .then((result) => {
        if (cancelled) return;
        if (result.memory.memory_id !== selectedId)
          throw new Error("wrong identity");
        setDetail(result);
        setTopic(
          result.memory.tags.slice(0, 2).join(", ").slice(0, 160) ||
            "Work discussed before",
        );
      })
      .catch(() => {
        if (!cancelled)
          setError(
            "This memory is not readable right now. Pick another one or refresh.",
          );
      })
      .finally(() => {
        if (!cancelled) setDetailLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  const available =
    detail !== null &&
    detail.memory.memory_id === selectedId &&
    !detailLoading &&
    guideDetailAvailable(detail, sources);
  const destinations = guideHosts(scene, detail, hosts);
  const host = destinations.find((item) => item.hostId === targetId);
  const canQuery = !!host?.installed && host.selfCheck?.state === "wired";
  const effectiveRoute = route === "query" && canQuery ? "query" : "context";
  const sourceIds = [
    ...new Set(
      detail?.evidence
        .map((item) => item.source_id)
        .filter((id): id is string => !!id),
    ),
  ];
  const nameOf = (id: string) =>
    sources.find((item) => item.source_id === id)?.display_name ??
    (id === "one-memory-manual" ? "Added in app" : id);
  const sourceLabel = sourceIds.map(nameOf).join(", ") || "Original app";
  const draft =
    available && detail
      ? draftForScenario(scene, effectiveRoute, topic, detail, sourceLabel)
      : "";
  const scenario =
    GUIDE_SCENARIOS.find((item) => item.id === scene) ?? GUIDE_SCENARIOS[0];
  const importedMessages = sources
    .filter((item) => item.authorized)
    .reduce((total, item) => total + (item.imported_messages ?? 0), 0);

  function changed() {
    actionEpoch.current++;
    setNotice("");
    setCopiedKey("");
    setReported(null);
    setBusy(false);
  }
  async function carry(open: boolean) {
    if (!detail || !draft || busy) return;
    const original = detail;
    const key = contextKey;
    const epoch = ++actionEpoch.current;
    const current = () =>
      mounted.current &&
      contextRef.current === key &&
      actionEpoch.current === epoch;
    setBusy(true);
    setNotice("");
    setReported(null);
    let validated = false;
    try {
      const [fresh, freshSources, freshHosts] = await Promise.all([
        getMemory(original.memory.memory_id),
        refreshSources(),
        effectiveRoute === "query" && window.oneMemory
          ? refreshRecallHosts()
          : Promise.resolve(hosts),
      ]);
      if (!current()) return;
      if (
        !sameGuideMemory(original, fresh) ||
        !guideDetailAvailable(fresh, freshSources)
      ) {
        setDetail(null);
        setCopiedKey("");
        setError(
          "The memory content or authorization has changed. Refresh and check again.",
        );
        return;
      }
      if (
        effectiveRoute === "query" &&
        !freshHosts.some(
          (item) =>
            item.hostId === host?.hostId &&
            item.present &&
            item.installed &&
            item.selfCheck?.state === "wired",
        )
      ) {
        setCopiedKey("");
        setNotice(
          "The connection status changed, so nothing was copied. Choose the method again.",
        );
        return;
      }
      validated = true;
      await copyText(draft);
      if (!current()) return;
      setCopiedKey(key);
      if (open && host?.handoff === "open" && window.oneMemory) {
        try {
          const result = await window.oneMemory.openHandoff(host.hostId, draft);
          if (current())
            setNotice(
              result.opened
                ? `Copied, and ${host.label} was asked to open. Not sent yet; check the input box.`
                : "Copied in full, but nothing was pre-filled. You can paste it in the target app; not sent yet.",
            );
        } catch {
          if (current())
            setNotice(
              "Copied in full, but the target app could not be opened. Not sent yet.",
            );
        }
      } else
        setNotice(
          "Copied in full. Not sent yet, and no new read authorization was added.",
        );
    } catch {
      if (current()) {
        if (!validated) {
          setDetail(null);
          setCopiedKey("");
          setError(
            "The current memory and authorization could not be verified, and the old preview was cleared. Refresh and retry.",
          );
        }
        setNotice(
          "Could not verify the latest content or finish copying. Retry; the target app was not opened.",
        );
      }
    } finally {
      if (current()) setBusy(false);
    }
  }

  return (
    <section className="view first-use-view" aria-labelledby="first-use-title">
      <header className="first-use-heading">
        <div>
          <p className="first-use-eyebrow">Get started</p>
          <h1 id="first-use-title">
            Make the next conversation pick up where this one ends
          </h1>
        </div>
        <button type="button" className="quiet" onClick={onClose}>
          <ArrowLeft size={16} aria-hidden="true" />
          Back to memories
        </button>
      </header>
      <fieldset className="first-use-scenarios">
        <legend className="sr-only">Choose a use scenario</legend>
        {GUIDE_SCENARIOS.map((item, index) => (
          <label key={item.id} data-selected={scene === item.id}>
            <input
              type="radio"
              name="guide-scene"
              value={item.id}
              checked={scene === item.id}
              onChange={() => {
                changed();
                setScene(item.id);
                setTargetId("");
                setRoute("context");
              }}
            />
            <span className="first-use-number">0{index + 1}</span>
            <span>{item.label}</span>
          </label>
        ))}
      </fieldset>
      <p className="first-use-description">{scenario.description}</p>
      {importedMessages > 0 && (
        <p className="first-use-receipt">
          This device has saved {importedMessages.toLocaleString()} raw messages
          ·{" "}
          {loading
            ? "Reading organized memories"
            : `This page has loaded ${memories.length} usable memories`}
          The two stages are counted separately.
        </p>
      )}
      <div className="first-use-path">
        <span>Pick a background</span>
        <ArrowRight size={14} aria-hidden="true" />
        <span>Check the sources</span>
        <ArrowRight size={14} aria-hidden="true" />
        <span>Try it in a new chat</span>
      </div>
      {error && (
        <p className="error-banner" role="alert">
          {error}
        </p>
      )}
      {loading ? (
        <p role="status">Reading existing memories on this device...</p>
      ) : !memories.length ? (
        <section className="first-use-empty">
          <FileText size={28} aria-hidden="true" />
          <h2>Leave behind one piece of background that really helps</h2>
          <p>
            There are no usable memories for this walkthrough yet. Imported
            source text is not the same as organized memories, and candidate
            content is not carried out automatically.
          </p>
          <div>
            <button type="button" onClick={onClose}>
              Go add a memory
            </button>
            <button type="button" className="secondary" onClick={onSources}>
              View imports and connections
            </button>
            <button
              type="button"
              className="quiet"
              onClick={() => setReload((n) => n + 1)}
            >
              <RefreshCw size={15} />
              Refresh
            </button>
          </div>
        </section>
      ) : (
        <div className="first-use-columns">
          <section
            className="first-use-background"
            aria-labelledby="guide-background-title"
          >
            <header>
              <h2 id="guide-background-title">The background used this time</h2>
              <button
                type="button"
                className="quiet"
                title="Refresh examples"
                aria-label="Refresh examples"
                onClick={() => {
                  changed();
                  setDetail(null);
                  setSelectedId("");
                  setReload((n) => n + 1);
                }}
              >
                <RefreshCw size={16} />
              </button>
            </header>
            <label className="first-use-field">
              Choose a memory
              <select
                value={selectedId}
                onChange={(event) => {
                  changed();
                  setSelectedId(event.target.value);
                }}
              >
                {memories.map((item) => (
                  <option key={item.memory_id} value={item.memory_id}>
                    {item.content.length > 90
                      ? `${item.content.slice(0, 90)}…`
                      : item.content}
                  </option>
                ))}
              </select>
            </label>
            <p className="first-use-meta">
              Chosen from the {memories.length} usable memories loaded here, not
              your full history.
            </p>
            {detailLoading ? (
              <p role="status">Checking the original evidence...</p>
            ) : available && detail ? (
              <>
                <blockquote className="first-use-memory">
                  {detail.memory.content}
                </blockquote>
                <p className="first-use-meta">
                  {sourceLabel} · Revision {detail.memory.revision} · Organized
                  automatically; needs your review
                </p>
                <p className="first-use-scope">
                  Scope:
                  {detail.memory.scopes.includes("all-apps")
                    ? "Set as available across apps; not a personal preference"
                    : detail.memory.scopes.includes("private")
                      ? "External memory recall is off; carrying it out manually needs separate confirmation"
                      : "Limited scope; the external recall range has not been widened"}
                </p>
                <details className="first-use-evidence" open>
                  <summary>
                    Original evidence · {detail.evidence.length} segments
                  </summary>
                  {detail.evidence.map((item) => (
                    <article key={item.evidence_id}>
                      <header>
                        <span>
                          {item.source_id
                            ? nameOf(item.source_id)
                            : "Unknown source"}
                        </span>
                        <span>
                          {item.written_by === "person"
                            ? "You"
                            : item.written_by === "agent"
                              ? "Agent"
                              : "Author not identified"}{" "}
                          ·{" "}
                          {item.created_at
                            ? new Date(item.created_at).toLocaleString()
                            : "Time unknown"}
                        </span>
                      </header>
                      <p>{item.verbatim}</p>
                    </article>
                  ))}
                </details>
              </>
            ) : (
              <p className="first-use-unavailable">
                The source text or current authorization is unavailable. Try
                another memory or refresh; this content will not be carried out.
              </p>
            )}
          </section>
          <section
            className="first-use-destination"
            aria-labelledby="guide-draft-title"
          >
            <h2 id="guide-draft-title">The opener for the next conversation</h2>
            <label className="first-use-field">
              Which app to try it in
              <select
                value={host?.hostId ?? ""}
                onChange={(event) => {
                  changed();
                  setTargetId(event.target.value);
                  setRoute("context");
                }}
              >
                <option value="">Pick an app yourself · copy only</option>
                {destinations.map((item) => (
                  <option key={item.hostId} value={item.hostId}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
            <p className="first-use-connection">
              {hostsError
                ? "The connection status cannot be read right now; retry or just copy the background."
                : !host
                  ? "No connected app selected yet. Carrying text in manually is not an installed connection."
                  : canQuery
                    ? "The last self-check reached it; whether this call used it correctly still needs checking."
                    : host.installed
                      ? "The connection is installed; read access is not verified yet."
                      : "No read connection installed."}
            </p>
            <button
              type="button"
              className="quiet first-use-connection-link"
              onClick={onSources}
            >
              <Link2 size={15} aria-hidden="true" />
              Manage app connections
            </button>
            <fieldset className="first-use-routes">
              <legend className="sr-only">Choose how to carry it in</legend>
              <label>
                <input
                  type="radio"
                  name="guide-route"
                  checked={effectiveRoute === "context"}
                  onChange={() => {
                    changed();
                    setRoute("context");
                  }}
                />
                Carry the background in
              </label>
              <label>
                <input
                  type="radio"
                  name="guide-route"
                  checked={effectiveRoute === "query"}
                  disabled={!canQuery}
                  onChange={() => {
                    changed();
                    setRoute("query");
                  }}
                />
                Let the app query
              </label>
            </fieldset>
            {effectiveRoute === "query" && (
              <label className="first-use-field">
                The topic to continue
                <input
                  value={topic}
                  maxLength={160}
                  onChange={(event) => {
                    changed();
                    setTopic(event.target.value);
                  }}
                />
              </label>
            )}
            <p className="first-use-meta">
              {effectiveRoute === "query"
                ? "Carries the query only; the expected answer above is not pushed into the new session."
                : scene === "new"
                  ? "Carries only the memory you chose; no personal profile is generated automatically."
                  : "Carries this memory and its citations in full; this is not an automatic recall test."}
            </p>
            <textarea
              className="first-use-draft"
              aria-label="Full opener preview"
              readOnly
              value={draft}
              placeholder="The opener appears once the original evidence is available"
              spellCheck={false}
            />
            <div className="first-use-actions">
              <button
                type="button"
                disabled={!draft || busy}
                onClick={() => void carry(false)}
              >
                <Copy size={16} aria-hidden="true" />
                {busy ? "Checking..." : "Copy opener"}
              </button>
              {host?.handoff === "open" && (
                <button
                  type="button"
                  className="secondary"
                  disabled={!draft || busy}
                  onClick={() => void carry(true)}
                >
                  <ExternalLink size={16} aria-hidden="true" />
                  Copy and open {host.label}
                </button>
              )}
            </div>
            {notice && (
              <p className="first-use-notice" role="status">
                {notice}
              </p>
            )}
            {copiedKey === contextKey && (
              <section className="first-use-result">
                <p>
                  No receipt for this external call yet. Did the new
                  conversation reference the right background?
                </p>
                <div>
                  <button
                    type="button"
                    className="quiet"
                    onClick={() => setReported("ok")}
                  >
                    <Check size={15} />I verified the result
                  </button>
                  <button
                    type="button"
                    className="quiet"
                    onClick={() => setReported("missing")}
                  >
                    Nothing was recalled
                  </button>
                </div>
                {reported && (
                  <p role="status">
                    {reported === "ok"
                      ? "Your feedback: verified. This is not an automatic acceptance record."
                      : "Your feedback: nothing was recalled. First check whether the source was imported, the memory scopes, and the app connection."}
                  </p>
                )}
              </section>
            )}
          </section>
        </div>
      )}
    </section>
  );
}

export function FirstUseInvitation({
  onOpen,
  onDismiss,
}: {
  onOpen(): void;
  onDismiss(): void;
}) {
  return (
    <aside className="first-use-invitation" aria-label="Getting-started invite">
      <span>
        Try carrying one of your own memories into the next conversation.
      </span>
      <button type="button" className="quiet" onClick={onOpen}>
        Try it once
        <ArrowRight size={15} aria-hidden="true" />
      </button>
      <button
        type="button"
        className="quiet"
        title="Show me later"
        aria-label="Close getting-started invite"
        onClick={onDismiss}
      >
        <X size={15} />
      </button>
    </aside>
  );
}
