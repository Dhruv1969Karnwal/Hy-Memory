/** Every claim on this page has to be traceable to docs/NARRATIVE.md.
 *
 * That file holds one column the page cannot show: the entry point that proves
 * each line. Copy edited here without checking it there is how a site starts
 * promising the six sources that are still fixture-only.
 */

const REFUSALS: ReadonlyArray<{ name: string; note: string }> = [
  {
    name: "No screenshots, no screen reading",
    note: "No screen recording and no screenshot permission",
  },
  {
    name: "No accessibility APIs",
    note: "Never reads other windows through system accessibility",
  },
  { name: "No OCR", note: "Never reads text out of images" },
  { name: "No keyboard or mouse logging", note: "No input monitoring" },
  { name: "No clipboard reading", note: "What you copy never enters memory" },
  { name: "No background recording", note: "No microphone capture" },
];

function Hero() {
  return (
    <header className="band hero">
      <div className="page">
        <p className="eyebrow">Local-first memory for your AI tools</p>
        <h1>Memory you can check</h1>
        <p className="lead">
          You have already told every AI your preferences, your judgement and
          the traps you fell into. MemHub gathers those words into one memory on
          this machine and lets other AI use it under your authorization -- and
          every memory leads back to the exact sentence it came from.
        </p>
        <div className="hero-actions">
          <a className="btn btn-primary" href="#download">
            Download for macOS
          </a>
          <a className="btn btn-quiet" href="#how">
            See how it works first
          </a>
        </div>
      </div>
    </header>
  );
}

function Evidence() {
  return (
    <section className="band" id="how">
      <div className="page">
        <p className="eyebrow">Why it can be checked</p>
        <h2>Extraction can be wrong; your words are never lost</h2>
        <div className="prose">
          <p>
            A memory system does not really fail by forgetting. It fails by
            remembering wrong with confidence, leaving you no way to verify.
            Models drift when they summarize you, and once they drift the
            original sentence is usually gone -- what remains is a line that
            sounds a lot like you but that you never said.
          </p>
          <p>
            So MemHub works in two steps: store the source text verbatim first,
            then extract memory from it. Extraction is extraction, source text
            is source text. When the model extracts wrongly you edit the
            extraction, not a fresh recollection; and if extraction fails
            completely, the sentence is still there.
          </p>
          <p>
            That is also why every memory carries its source. It is not a
            summary you are asked to trust; it is a record you can look up and
            compare at any time.
          </p>
        </div>
      </div>
    </section>
  );
}

function Refusals() {
  return (
    <section className="band" id="privacy">
      <div className="page">
        <p className="eyebrow">What we do not do</p>
        <h2>It only reads conversations you already wrote down</h2>
        <p className="lead">
          How much can be read depends on how it is read. We use none of the six
          ways below, and that is not a promise but a gate every commit has to
          pass.
        </p>
        <div className="refusals">
          {REFUSALS.map((item) => (
            <div className="refusal" key={item.name}>
              <div className="refusal-name">{item.name}</div>
              <p className="refusal-note">{item.note}</p>
            </div>
          ))}
        </div>
        <p className="gate">
          <code>scripts/verify_zero_collection.py</code>
          <span>
            Scans three layers: source, dependencies and system permissions; a
            failure keeps it out of the trunk.
          </span>
        </p>
      </div>
    </section>
  );
}

export function App() {
  return (
    <main>
      <Hero />
      <Evidence />
      <Refusals />
    </main>
  );
}
