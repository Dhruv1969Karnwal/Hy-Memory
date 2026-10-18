"""
Entity extraction (ported from mem0 OSS `mem0/utils/entity_extraction.py`).

Extracts four kinds of entity from text:
- PROPER:   proper noun sequences (person/place/brand names, consecutive capitalised words)
- QUOTED:   text inside quotes (titles/specific terms)
- COMPOUND: compound noun phrases (e.g. "machine learning")
- NOUN:     a single noun recovered as fallback from adverbial compound patterns

Public API:
    extract_entities(text) -> List[Tuple[entity_type, entity_text]]

Design notes:
- Uses the spaCy **full pipeline** (including tagger/parser, unlike the stripped variant disabled in lemmatize.py),
  because entity extraction depends on POS / dep / noun_chunks.
- When spaCy is unavailable (not installed or the model is missing) it returns [], and the entity channel degrades as a whole (same as mem0).
- With Chinese text the English spaCy model barely extracts any entity, which is expected (mem0 behaves the same).
"""

from __future__ import annotations

import logging
import re
import threading
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ========================================================================
# Lazy loading of the spaCy full pipeline (separate from the stripped pipeline in lemmatize.py)
# ========================================================================

_nlp_full = None
_load_failed = False
_lock = threading.Lock()


def get_nlp_full():
    """Return the spaCy model with the full pipeline (tagger/parser/NER) used for entity extraction.

    Returns None when spaCy is not installed or the en_core_web_sm model is missing (callers degrade to no entities).
    """
    global _nlp_full, _load_failed
    if _load_failed:
        return None
    if _nlp_full is not None:
        return _nlp_full
    with _lock:
        if _nlp_full is not None:
            return _nlp_full
        if _load_failed:
            return None
        try:
            import spacy
            _nlp_full = spacy.load("en_core_web_sm")
            logger.info("[entities] spaCy en_core_web_sm full pipeline loaded")
        except Exception as e:
            logger.warning(
                f"[entities] spaCy full model unavailable, entity extraction disabled: {e}"
            )
            _load_failed = True
            return None
    return _nlp_full


# ========================================================================
# Word lists (ported from mem0)
# ========================================================================

_GENERIC_HEADS = {
    "thing", "stuff", "way", "time", "experience", "situation", "case",
    "fact", "matter", "issue", "idea", "thought", "feeling", "place",
    "area", "part", "kind", "type", "sort", "lot", "bit", "day", "year",
    "week", "month", "moment", "instance", "example", "technique",
    "method", "approach", "process", "step", "tool", "result", "outcome",
    "goal", "task", "item", "topic", "scale", "size", "level", "degree",
    "amount", "number", "style", "look", "color", "colour", "shape",
    "form", "piece", "section", "side", "end", "edge", "surface", "point",
}

_CIRCUMSTANTIAL_MODS = {
    "solo", "individual", "team", "group", "joint", "collaborative",
    "first", "last", "next", "previous", "final", "initial", "main", "side",
}

_NON_SPECIFIC_ADJ = {
    "many", "few", "several", "some", "any", "all", "most", "more",
    "less", "much", "little", "enough", "various", "numerous", "multiple",
    "countless", "great", "good", "bad", "nice", "terrible", "awful",
    "awesome", "amazing", "wonderful", "horrible", "excellent", "poor",
    "best", "worst", "fine", "okay", "new", "old", "recent", "past",
    "future", "current", "previous", "next", "last", "first", "latest",
    "early", "late", "former", "modern", "ancient", "big", "small",
    "large", "tiny", "huge", "enormous", "long", "short", "tall", "high",
    "low", "wide", "narrow", "thick", "thin", "deep", "shallow",
    "similar", "different", "same", "other", "another", "such", "certain",
    "important", "main", "major", "minor", "key", "primary", "real",
    "actual", "true", "whole", "entire", "full", "complete", "total",
    "basic", "simple", "interesting", "boring", "exciting", "special",
    "particular", "general", "common", "unique", "rare", "typical",
    "usual", "normal", "regular", "possible", "likely", "potential",
    "available", "necessary", "only", "solo", "individual", "team",
    "group", "joint", "collaborative", "final", "initial", "side",
}

_GENERIC_ENDINGS = {
    "work", "works", "job", "jobs", "task", "tasks", "stuff", "things",
    "thing", "info", "information", "details", "data", "content",
    "material", "materials", "activities", "activity", "efforts", "effort",
    "options", "option", "choices", "choice", "results", "result",
    "output", "outputs", "products", "product", "items", "item",
}

_GENERIC_CAPS = {
    "works", "items", "things", "stuff", "resources", "options", "tips",
    "ideas", "steps", "ways", "methods", "tools", "features", "benefits",
    "examples", "details", "notes", "instructions", "guidelines",
    "recommendations", "suggestions", "overview", "summary", "conclusion",
    "introduction", "pros", "cons", "advantages", "disadvantages",
}

_FORMATTING_MARKERS = {"*", "-", "+", "•", "–", "—", "#", "##", "###", "**", "__"}


# ========================================================================
# Internal helpers (ported from mem0)
# ========================================================================

def _is_sentence_start(tokens: list, idx: int) -> bool:
    if idx == 0:
        return True
    tok = tokens[idx]
    if tok.is_sent_start:
        return True
    prev = tokens[idx - 1].text
    return prev in ".!?:" or prev in _FORMATTING_MARKERS or "\n" in prev


def _strip_generic_ending(toks: list) -> list:
    if len(toks) <= 1:
        return toks
    last = toks[-1].lemma_.lower() if hasattr(toks[-1], "lemma_") else toks[-1].lower()
    return toks[:-1] if last in _GENERIC_ENDINGS and len(toks) > 2 else toks


def _lemmatize_compound(toks: list) -> str:
    return " ".join(t.lemma_ if t.pos_ == "NOUN" else t.text for t in toks)


def _has_artifacts(txt: str) -> bool:
    return any(
        [
            "**" in txt or "__" in txt or ":*" in txt,
            re.search(r"\s\*\s|\s\*$|^\*\s", txt),
            "  " in txt or "\n" in txt or "\t" in txt,
            len(txt) > 100,
            txt.startswith(("•", "-", "+", "–", "—")),
        ]
    )


# ========================================================================
# Public API
# ========================================================================

def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Extract entities from text, returning a deduplicated list of (entity_type, entity_text).

    entity_type ∈ {PROPER, QUOTED, COMPOUND, NOUN}. Returns [] when spaCy is unavailable.
    """
    if not text or not text.strip():
        return []
    nlp = get_nlp_full()
    if nlp is None:
        return []
    try:
        doc = nlp(text)
    except Exception as e:
        logger.debug(f"[entities] spaCy parse failed: {e}")
        return []
    return _extract_entities_from_doc(doc)


def _extract_entities_from_doc(doc) -> List[Tuple[str, str]]:
    entities: List[Tuple[str, str]] = []
    text = doc.text
    tokens = list(doc)

    # === PROPER NOUN SEQUENCES ===
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.text in _FORMATTING_MARKERS:
            i += 1
            continue
        is_cap = tok.text and tok.text[0].isupper()
        is_label = i + 1 < len(tokens) and tokens[i + 1].text == ":"

        if is_cap and not is_label and tok.pos_ in {"PROPN", "NOUN", "ADJ"}:
            seq = [(tok, i)]
            j = i + 1
            while j < len(tokens):
                t = tokens[j]
                if (t.text and t.text[0].isupper()) or t.text.lower() in {
                    "'s", "of", "the", "in", "and", "for", "at", "is",
                }:
                    seq.append((t, j))
                    j += 1
                else:
                    break
            while seq and seq[-1][0].text.lower() in {"of", "the", "in", "and", "for", "at", "is", "'s"}:
                seq.pop()
            if seq:
                has_mid_cap = any(
                    not _is_sentence_start(tokens, idx)
                    for (t, idx) in seq
                    if t.text[0].isupper() and t.text.lower() not in {"'s", "of", "the", "in", "and", "for", "at", "is"}
                )
                if has_mid_cap:
                    phrase = "".join(t.text_with_ws for (t, idx) in seq).strip()
                    if len(phrase) > 2:
                        entities.append(("PROPER", phrase))
            i = j
        else:
            i += 1

    # === QUOTED TEXT ===
    for m in re.finditer(r'"([^"]+)"', text):
        if len(m.group(1).strip()) > 2:
            entities.append(("QUOTED", m.group(1).strip()))
    for m in re.finditer(r"(?:^|[\s\(\[{,;])'([^']+)'(?=[\s\.,;:!?\)\]]|$)", text):
        if len(m.group(1).strip()) > 2:
            entities.append(("QUOTED", m.group(1).strip()))

    # === NOUN-NOUN COMPOUNDS ===
    for chunk in doc.noun_chunks:
        chunk_tokens = list(chunk)
        split_indices: list = []
        poss_splits: list = []
        for idx, tok in enumerate(chunk_tokens):
            if tok.dep_ == "case" and tok.text in {"'s", "’s", "'"}:
                split_indices.append(idx)
                poss_splits.append(idx)
            elif tok.pos_ == "PUNCT" and tok.text in {"'", '"', "‘", "’", "“", "”"}:
                split_indices.append(idx)

        if split_indices:
            groups: list = []
            prev = 0
            for split_idx in split_indices:
                if split_idx > prev:
                    groups.append(chunk_tokens[prev:split_idx])
                if split_idx in poss_splits:
                    next_split = next((s for s in split_indices if s > split_idx), None)
                    owned = chunk_tokens[split_idx + 1: next_split if next_split else len(chunk_tokens)]
                    if owned:
                        first_content = next((t for t in owned if t.pos_ not in {"PUNCT", "PART"}), None)
                        if not (first_content and first_content.text and first_content.text[0].isupper()):
                            prev = next_split if next_split else len(chunk_tokens)
                            continue
                prev = split_idx + 1
            if prev < len(chunk_tokens):
                groups.append(chunk_tokens[prev:])
        else:
            groups = [chunk_tokens]

        for group in groups:
            if not group:
                continue
            head = next((t for t in reversed(group) if t.pos_ in {"NOUN", "PROPN"}), None)
            if not head:
                continue
            head_generic = head.lemma_.lower() in _GENERIC_HEADS
            content = [
                t
                for t in group
                if t.pos_ not in {"DET", "PRON", "PUNCT", "PART", "ADP", "SCONJ", "NUM"} and (t.pos_ == "ADJ" or not t.is_stop)
            ]
            if not content:
                continue

            compound_toks = [t for t in content if t.dep_ == "compound"]
            adj_toks = [t for t in content if t.pos_ == "ADJ" or t.dep_ == "amod"]
            has_spec_adj = any(t.lemma_.lower() not in _NON_SPECIFIC_ADJ for t in adj_toks)
            if head_generic and not has_spec_adj and not compound_toks:
                continue

            if compound_toks:
                is_circ = any(t.lemma_.lower() in _CIRCUMSTANTIAL_MODS for t in compound_toks)
                if is_circ:
                    val = head.lemma_ if head.pos_ == "NOUN" else head.text
                    if len(val) > 2:
                        entities.append(("NOUN", val))
                else:
                    filtered = _strip_generic_ending(
                        [t for t in content if not (t.pos_ == "ADJ" and t.lemma_.lower() in _NON_SPECIFIC_ADJ)]
                    )
                    if filtered:
                        phrase = _lemmatize_compound(filtered)
                        if len(phrase) > 3 and " " in phrase:
                            entities.append(("COMPOUND", phrase))
            elif len(content) > 1 and has_spec_adj:
                filtered = _strip_generic_ending(
                    [t for t in content if not ((t.pos_ == "ADJ" or t.dep_ == "amod") and t.lemma_.lower() in _NON_SPECIFIC_ADJ)]
                )
                if filtered:
                    phrase = _lemmatize_compound(filtered)
                    if len(phrase) > 3 and " " in phrase:
                        entities.append(("COMPOUND", phrase))

    # === FALLBACK: Mis-tagged VERB heads ===
    processed = {e[1].lower() for e in entities if e[0] == "COMPOUND"}
    generic_verb_heads = _GENERIC_HEADS | {"find", "buy", "purchase", "sale", "deal", "trip", "visit"}

    def collect_compounds(head):
        return [t for t in doc if t.head == head and t.dep_ == "compound"]

    for tok in doc:
        if tok.pos_ == "VERB" and tok.dep_ in {"pobj", "dobj", "nsubj"}:
            comps = sorted(collect_compounds(tok), key=lambda t: t.i)
            if comps:
                phrase_toks = comps if tok.lemma_.lower() in generic_verb_heads else comps + [tok]
                phrase = " ".join(t.text for t in phrase_toks)
                if phrase.lower() not in processed and len(phrase) > 3 and " " in phrase:
                    entities.append(("COMPOUND", phrase))
                    processed.add(phrase.lower())

    # === DEDUPLICATION & CLEANUP ===
    seen: set = set()
    deduped = []
    for t, e in entities:
        k = e.lower().strip()
        if k not in seen and len(k) > 2:
            seen.add(k)
            deduped.append((t, e))

    cleaned: List[Tuple[str, str]] = []
    for etype, etext in deduped:
        txt = re.sub(r"^\*+\s*|\s*\*+$", "", etext.strip())
        txt = re.sub(r"\s*:+$", "", txt)
        txt = re.sub(r"^\d+\s*\.\s*", "", txt)
        if not txt or len(txt) <= 2 or _has_artifacts(txt):
            continue
        if etype == "PROPER" and " " not in txt and txt.lower() in _GENERIC_CAPS:
            continue
        cleaned.append((etype, txt))

    # Keep best type per entity (PROPER > COMPOUND > QUOTED > NOUN)
    type_pri = {"PROPER": 0, "COMPOUND": 1, "QUOTED": 2, "NOUN": 3, "VERB": 4}
    best: dict = {}
    for t, e in cleaned:
        k = e.lower()
        if k not in best or type_pri.get(t, 99) < type_pri.get(best[k][0], 99):
            best[k] = (t, e)
    deduped = list(best.values())

    # Remove entities that are substrings of longer entities
    all_lower = [e[1].lower() for e in deduped]
    return [(t, e) for t, e in deduped if not any(e.lower() != o and e.lower() in o for o in all_lower)]
