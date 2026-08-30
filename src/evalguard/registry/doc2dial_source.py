"""Parses the raw Doc2Dial corpus into plain candidate records.

Doc2Dial (Feng et al., EMNLP 2020) is a goal-oriented, document-grounded dialogue
dataset covering four US government-service domains (ssa, va, dmv, studentaid). This
module reads the *raw* release files directly — not the HuggingFace `datasets` loading
script — because that script requires `trust_remote_code` and isn't schema-stable
across `datasets` versions. Fetch the raw files once with
`scripts/build_doc2dial_dataset.py --download`; everything below just parses what's on
disk.

Source: https://doc2dial.github.io/file/doc2dial_v1.0.1.zip

Verified against the actual downloaded JSON (not assumed from the HF dataset card):

- `doc2dial_doc.json` → `{"doc_data": {domain: {doc_id: {title, doc_id, domain,
  doc_text, spans, doc_html_ts, doc_html_raw}}}}`. `spans` is a **dict keyed by
  `sp_id`** (the HF dataset card's inferred parquet schema flattens this to a list —
  the raw release does not; do not assume the card's shape).
- `doc2dial_dial_{train,validation,test}.json` → `{"dial_data": {domain: {doc_id:
  [dialogue, ...]}}}`. Each dialogue is `{dial_id, doc_id, domain, turns}`; each turn is
  `{turn_id, role, da, references, utterance}` where `references` is a list of
  `{sp_id, label}` pointing into that doc's `spans`.
- Dialogue acts observed in the validation split: `query_condition`, `respond_solution`,
  `query_solution`, `response_positive`, `response_negative`,
  `respond_solution_positive`, `respond_solution_negative`. No turn in this release has
  empty `references` — i.e. Doc2Dial does **not** ship an "unanswerable" label. The
  abstention category (case_builder.py) is therefore derived (domain-mismatched
  question/context pairing), not sourced from a native Doc2Dial field — documented
  there, not assumed here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GroundedCandidate:
    """A real (question, grounding context, answer) triple extracted from one dialogue turn.

    `dial_id` + `turn_id` uniquely identifies the source turn — used to guarantee the
    "no overlap" requirement when candidates are partitioned across eval categories.
    """

    dial_id: str
    turn_id: int
    domain: str
    doc_id: str
    question: str
    context: str
    reference_answer: str

    @property
    def source_key(self) -> tuple[str, int]:
        return (self.dial_id, self.turn_id)


# Agent dialogue acts that represent a real, content-bearing answer grounded in spans.
_ANSWER_DIALOGUE_ACTS = frozenset({"respond_solution", "respond_solution_positive"})

# Skip degenerate questions like "yes" / "ok, thanks" that carry no real content.
_MIN_QUESTION_WORDS = 4

# Grounding passages shorter than this are usually a bare section heading (e.g. "Railroad
# Earnings") reconstructed from a single short span, not a usable context — found by
# manual audit of the dev sample (docs/ARCHITECTURE.md hardening pass).
_MIN_CONTEXT_WORDS = 6

# Multi-turn dialogue turns taken out of context are sometimes a continuation of the
# previous utterance ("and how can i pay it?", "ok and that is the only condition?",
# "Yes, that's what I just said.") rather than a self-contained question — found by
# manual dev-sample audit. "yes"/"no" are included because a bare affirmation/denial is
# only meaningful as a reply to an unstated prior yes/no question — it carries no
# question of its own. Filtered by first token, case-insensitive.
_FRAGMENT_STARTERS = frozenset(
    {
        "and",
        "but",
        "so",
        "or",
        "also",
        "then",
        "because",
        "ok",
        "okay",
        "yeah",
        "yep",
        "well",
        "yes",
        "no",
    }
)

# A handful of Doc2Dial agent turns are generic transition/punt phrases rather than a
# real answer to the preceding question ("Here's some additional information...", "I
# suggest you check with...") — found by manual dev-sample audit.
_FILLER_ANSWER_RE = re.compile(
    r"^(here'?s\s+some|here\s+is\s+some|i\s+suggest\s+you|let\s+me\s+know\s+if"
    r"|thanks?\s+for|sorry,?\s+(but\s+)?(i|we)\b)",
    re.IGNORECASE,
)

# Short questions ending in a bare, objectless verb ("How can I request?") are vague to
# the point of being untestable — found by manual dev-sample audit. Only applied to
# short questions so a longer, genuinely self-contained question ending in the same word
# ("...call the veterans crisis line for help?") isn't penalized.
_VAGUE_TRAILING_VERBS = frozenset({"request", "apply", "qualify", "help", "proceed", "continue"})
_VAGUE_QUESTION_MAX_WORDS = 5

# Doc2Dial's own utterance/span text occasionally carries tokenizer-artifact spacing —
# a space before a contraction or punctuation mark ("Here 's some ... what 's right for
# you :") — inherited from how a handful of source dialogues were authored. Collapsing it
# is a pure text-normalization fix; it changes no content.
_DETOK_CONTRACTION_RE = re.compile(r"\s+('(?:s|re|t|ve|ll|d|m))\b", re.IGNORECASE)
_DETOK_PUNCTUATION_RE = re.compile(r"\s+([,.:;!?%])")


def _normalize_text(text: str) -> str:
    """Collapse tokenizer-artifact spacing and excess whitespace. See module docstring."""
    text = _DETOK_CONTRACTION_RE.sub(r"\1", text)
    text = _DETOK_PUNCTUATION_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_fragment_question(question: str) -> bool:
    first_word = re.split(r"\W+", question, maxsplit=1)[0].lower()
    return first_word in _FRAGMENT_STARTERS


def _is_filler_answer(answer: str) -> bool:
    return bool(_FILLER_ANSWER_RE.match(answer))


def _is_vague_question(question: str) -> bool:
    words = question.split()
    if len(words) > _VAGUE_QUESTION_MAX_WORDS:
        return False
    last_word = re.sub(r"\W+$", "", words[-1]).lower() if words else ""
    return last_word in _VAGUE_TRAILING_VERBS


def load_doc_json(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load `doc2dial_doc.json` → `{domain: {doc_id: doc_record}}`."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw["doc_data"]


def load_dial_json(path: str | Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Load a `doc2dial_dial_*.json` split → `{domain: {doc_id: [dialogue, ...]}}`."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw["dial_data"]


def reconstruct_context(doc: dict[str, Any], sp_ids: list[str]) -> str:
    """Rebuild a grounding passage from a turn's referenced span ids.

    Spans are concatenated in document order (by `start_sp`), not reference order,
    since `references` on a turn can list spans out of document sequence.
    """
    spans = doc["spans"]
    matched = (spans[sp_id] for sp_id in sp_ids if sp_id in spans)
    ordered = sorted(matched, key=lambda s: s["start_sp"])
    return " ".join(s["text_sp"].strip() for s in ordered).strip()


def extract_grounded_candidates(
    docs: dict[str, dict[str, Any]],
    dialogues: dict[str, dict[str, list[dict[str, Any]]]],
) -> list[GroundedCandidate]:
    """Walk every dialogue and pull out (user question -> grounded agent answer) pairs.

    A candidate requires: an agent turn with a content-bearing dialogue act and
    non-empty `references`, immediately preceded by a non-trivial user turn. Both
    conditions are checked directly against the raw turn data — nothing here is
    inferred from the HF card.

    Text is normalized (tokenizer-artifact spacing collapsed) and five quality filters
    are applied before a candidate is accepted — all added after a manual dev-sample
    audit turned up each failure mode by name (see docs/ARCHITECTURE.md hardening pass):
    too-short question, fragment/continuation question, vague objectless question,
    filler/non-answer, and too-thin context. Corpus size (2800+ eligible turns) gives
    plenty of headroom to filter conservatively rather than keep a borderline case.
    """
    candidates: list[GroundedCandidate] = []

    for domain, doc_map in dialogues.items():
        for doc_id, dialogue_list in doc_map.items():
            doc = docs.get(domain, {}).get(doc_id)
            if doc is None:
                continue
            for dialogue in dialogue_list:
                turns = dialogue["turns"]
                for i, turn in enumerate(turns):
                    if turn["role"] != "agent" or turn["da"] not in _ANSWER_DIALOGUE_ACTS:
                        continue
                    if not turn["references"]:
                        continue
                    if i == 0 or turns[i - 1]["role"] != "user":
                        continue

                    question = _normalize_text(turns[i - 1]["utterance"])
                    if len(question.split()) < _MIN_QUESTION_WORDS:
                        continue
                    if _is_fragment_question(question):
                        continue
                    if _is_vague_question(question):
                        continue

                    answer = _normalize_text(turn["utterance"])
                    if _is_filler_answer(answer):
                        continue

                    context = _normalize_text(
                        reconstruct_context(doc, [ref["sp_id"] for ref in turn["references"]])
                    )
                    if len(context.split()) < _MIN_CONTEXT_WORDS:
                        continue

                    candidates.append(
                        GroundedCandidate(
                            dial_id=dialogue["dial_id"],
                            turn_id=turn["turn_id"],
                            domain=domain,
                            doc_id=doc_id,
                            question=question,
                            context=context,
                            reference_answer=answer,
                        )
                    )
    return candidates
