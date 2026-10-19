import type { GovernedMemory, MemoryDetail } from "@one-memory/contracts";
import { countLabel } from "./count-label";

export function MatterMemoryPanel({
  memories,
  selectedMemoryId,
  detail,
  loading,
  onSelect,
}: {
  memories: GovernedMemory[];
  selectedMemoryId: string | null;
  detail: MemoryDetail | null;
  loading: boolean;
  onSelect(memoryId: string): void;
}) {
  return (
    <section
      className="matter-memory-panel"
      aria-label="Matter baseline memories"
    >
      <header>
        <h3>Baseline memories</h3>
        <span>{memories.length}</span>
      </header>
      <div className="matter-memory-list">
        {memories.map((memory) => (
          <button
            type="button"
            key={memory.memory_id}
            aria-label={`Select memory ${memory.content}`}
            aria-pressed={selectedMemoryId === memory.memory_id}
            onClick={() => onSelect(memory.memory_id)}
          >
            <span>{memory.content}</span>
            <small>
              Baseline memory ·{" "}
              {countLabel(memory.evidence_ids.length, "quote")} · r
              {memory.revision}
            </small>
          </button>
        ))}
      </div>
      {loading && <p role="status">Loading baseline memories and quotes...</p>}
      {detail && !loading && (
        <section
          className="matter-memory-detail"
          aria-label="Baseline memory originals"
        >
          <h3>Source quotes</h3>
          {detail.evidence.map((item) => (
            <section key={item.evidence_id} className="matter-memory-excerpt">
              <p>
                {item.source_id ?? "Unknown source"} ·{" "}
                {item.written_by === "person"
                  ? "User quote"
                  : item.written_by === "agent"
                    ? "Agent message"
                    : "Unknown author"}
              </p>
              <small>
                {item.created_at
                  ? new Date(item.created_at).toLocaleString("zh-CN")
                  : "Message time unknown"}
              </small>
              <blockquote>
                {item.available
                  ? item.verbatim
                  : "The source quote is no longer readable"}
              </blockquote>
            </section>
          ))}
        </section>
      )}
    </section>
  );
}
