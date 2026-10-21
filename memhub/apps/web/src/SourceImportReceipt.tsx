import type {
  ConversationTranscript,
  SourceImportReceipt as Receipt,
  SourceResponse,
} from "@one-memory/contracts";
import { FileText, RefreshCw, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { getConversation, getSourceImportReceipt } from "./api";
import { groupedCountLabel } from "./count-label";

export function SourceImportReceipt({ source }: { source: SourceResponse }) {
  return source.authorized ? (
    <ReceiptBody key={source.source_id} source={source} />
  ) : null;
}

function ReceiptBody({ source }: { source: SourceResponse }) {
  const [receipt, setReceipt] = useState<Receipt | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [opened, setOpened] = useState(false);
  const [transcript, setTranscript] = useState<ConversationTranscript | null>(
    null,
  );
  const [reading, setReading] = useState(false);
  const [readError, setReadError] = useState("");
  const generation = useRef(0);
  const conversationGeneration = useRef(0);
  useEffect(
    () => () => {
      generation.current++;
      conversationGeneration.current++;
    },
    [],
  );

  async function load() {
    const current = ++generation.current;
    conversationGeneration.current++;
    setReceipt(null);
    setTranscript(null);
    setReadError("");
    setError("");
    setLoading(true);
    setReading(false);
    try {
      const result = await getSourceImportReceipt(source.source_id);
      if (generation.current !== current) return;
      if (result.source_id !== source.source_id)
        throw new Error("source mismatch");
      setReceipt(result);
    } catch {
      if (generation.current === current)
        setError(
          "Saved content cannot be read right now and the old receipt was collapsed. Check the authorization or retry.",
        );
    } finally {
      if (generation.current === current) setLoading(false);
    }
  }

  async function read(evidenceId: string) {
    const current = ++conversationGeneration.current;
    setTranscript(null);
    setReadError("");
    setReading(true);
    try {
      const result = await getConversation(evidenceId);
      if (conversationGeneration.current !== current) return;
      if (
        result.source_id !== source.source_id ||
        !result.messages.some((item) => item.evidence_id === evidenceId)
      )
        throw new Error("conversation mismatch");
      setTranscript(result);
    } catch {
      if (conversationGeneration.current === current) {
        setReceipt(null);
        setReadError(
          "This source text cannot be read; its authorization may have been revoked. Refresh the receipt.",
        );
      }
    } finally {
      if (conversationGeneration.current === current) setReading(false);
    }
  }

  return (
    <details
      className="source-import-receipt"
      onToggle={(event) => {
        const next = event.currentTarget.open;
        setOpened(next);
        if (next) void load();
        else {
          generation.current++;
          conversationGeneration.current++;
          setReceipt(null);
          setTranscript(null);
          setLoading(false);
          setReading(false);
        }
      }}
    >
      <summary>View read content</summary>
      {opened && (
        <div className="source-receipt-body">
          <header>
            <span>Saved source text · {source.display_name}</span>
            <button
              type="button"
              className="quiet"
              title="Refresh read content"
              aria-label="Refresh read content"
              disabled={loading}
              onClick={() => void load()}
            >
              <RefreshCw size={15} />
            </button>
          </header>
          {loading && <p role="status">Checking locally saved records...</p>}
          {error && <p role="alert">{error}</p>}
          {receipt && (
            <>
              <dl className="source-receipt-counts">
                <div>
                  <dt>Original messages</dt>
                  <dd>{receipt.messages.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>Conversations</dt>
                  <dd>{receipt.conversations.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>User messages</dt>
                  <dd>{receipt.person_messages.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>Agent messages</dt>
                  <dd>{receipt.agent_messages.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>Unknown author</dt>
                  <dd>{receipt.unknown_author_messages.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>Unknown original time</dt>
                  <dd>{receipt.unknown_time_messages.toLocaleString()}</dd>
                </div>
                {receipt.tool_messages > 0 && (
                  <div>
                    <dt>Tool records</dt>
                    <dd>{receipt.tool_messages.toLocaleString()}</dd>
                  </div>
                )}
              </dl>
              <p className="source-receipt-note">
                {receipt.unknown_time_messages > 0
                  ? `${groupedCountLabel(receipt.unknown_time_messages, "record")} have no original time, the import time was not substituted.`
                  : "Original times come from the saved records."}{" "}
                Author counts are classified from the current records; they do
                not mean extraction finished or that meaning was verified.
              </p>
              <p className="source-receipt-meta">
                Measured at {time(receipt.measured_at)}
                {receipt.messages !== source.imported_messages
                  ? " · import counts are still updating, refresh to check"
                  : ""}
              </p>
              {receipt.examples.length > 0 ? (
                <section
                  className="source-receipt-examples"
                  aria-label="Samples of saved source text"
                >
                  <h3>Samples of saved source text</h3>
                  <p className="source-receipt-meta">
                    At most 3 samples are shown; they are not the whole history
                    or a recommended memory.
                  </p>
                  {receipt.examples.map((example) => (
                    <article key={example.evidence_id}>
                      <header>
                        <span>
                          {author(example.written_by)} ·{" "}
                          {time(example.created_at)}
                        </span>
                        <button
                          type="button"
                          className="quiet"
                          onClick={() => void read(example.evidence_id)}
                        >
                          <FileText size={14} aria-hidden="true" />
                          View full conversation
                        </button>
                      </header>
                      <p className="source-receipt-quote">{example.excerpt}</p>
                      {example.truncated && (
                        <small>
                          The source text is long; only the first 600 characters
                          are shown here.
                        </small>
                      )}
                    </article>
                  ))}
                </section>
              ) : (
                <p>This source has no saved source text yet.</p>
              )}
            </>
          )}
          {reading && <p role="status">Reading the full conversation...</p>}
          {readError && <p role="alert">{readError}</p>}
          {transcript && (
            <section
              className="source-receipt-conversation"
              aria-label="Full conversation"
            >
              <header>
                <h3>Full conversation</h3>
                <button
                  type="button"
                  className="quiet"
                  title="Close the full conversation"
                  aria-label="Close the full conversation"
                  onClick={() => {
                    conversationGeneration.current++;
                    setTranscript(null);
                  }}
                >
                  <X size={15} />
                </button>
              </header>
              <ol>
                {transcript.messages.map((message) => (
                  <li key={message.evidence_id}>
                    <small>
                      {author(message.written_by)} · {time(message.created_at)}
                    </small>
                    <p>{message.verbatim}</p>
                  </li>
                ))}
              </ol>
            </section>
          )}
        </div>
      )}
    </details>
  );
}

function author(value: string) {
  return value === "person"
    ? "User"
    : value === "agent"
      ? "Agent"
      : value === "tool"
        ? "Tool"
        : "Unknown author";
}
function time(value: string | null) {
  return value && Number.isFinite(Date.parse(value))
    ? new Date(value).toLocaleString()
    : "Unknown time";
}
