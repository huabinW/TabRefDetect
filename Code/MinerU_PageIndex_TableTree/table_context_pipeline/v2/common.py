"""Shared helpers for the scope-aware candidate inventory workflow."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from popo_workflow_helpers import (  # noqa: E402
    extract_popo_nodes,
    match_table_anchors,
)


REFERENCE_TITLE_RE = re.compile(
    r"^\s*(?:(?:\d+(?:\.\d+)*)|[A-Z])?[.)]?\s*"
    r"(?:references?|bibliography|works\s+cited)\s*$",
    re.I,
)
TABLE_PREFIX_RE = re.compile(r"\b(?:tables?|tabs?|tbls?)\.?\s*", re.I)
DESIGNATOR_RE = re.compile(
    r"(?:[A-Za-z]?\s*\d+[A-Za-z]?|[IVXLCDM]+)(?:\s*\([A-Za-z0-9]+\))?",
    re.I,
)
CONNECTOR_RE = re.compile(
    r"\s*(?P<connector>,(?:\s*(?:and|&))?|\band\b|&|\bto\b|[-\u2013\u2014])\s*",
    re.I,
)
POPO_SEPARATOR_RE = re.compile(r"<\|(?:txt_)?split\|>|\n\s*\n", re.I)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.+@/-]{1,}")

ABBREVIATIONS = {
    "al.",
    "approx.",
    "cf.",
    "dr.",
    "e.g.",
    "eq.",
    "eqs.",
    "etc.",
    "fig.",
    "figs.",
    "i.e.",
    "mr.",
    "mrs.",
    "ms.",
    "no.",
    "prof.",
    "sec.",
    "secs.",
    "st.",
    "tab.",
    "tbl.",
    "vs.",
}

CONDITION_HINT_PATTERNS = {
    "dataset": re.compile(r"\b(?:dataset|corpus|benchmark|train(?:ing)? set|test set|validation set)\b", re.I),
    "model": re.compile(r"\b(?:model|encoder|decoder|architecture|parameter(?:s|ization)?)\b", re.I),
    "metric": re.compile(r"\b(?:metric|accuracy|precision|recall|f[- ]?1|bleu|rouge|auc|pass@\d+)\b", re.I),
    "prompt": re.compile(r"\b(?:prompt|instruction|template|demonstration|few[- ]shot|zero[- ]shot|\d+[- ]shot)\b", re.I),
    "split": re.compile(r"\b(?:split|fold|cross[- ]validation|held[- ]out|dev(?:elopment)? set)\b", re.I),
    "baseline": re.compile(r"\b(?:baseline|comparison|compared with|control group)\b", re.I),
    "training": re.compile(r"\b(?:train|finetun|pretrain|optimizer|learning rate|batch size|epoch|warmup|weight decay)\b", re.I),
    "decoding": re.compile(r"\b(?:decod|beam size|temperature|top[- ]?[kp]|sampling|greedy)\b", re.I),
    "evaluation": re.compile(r"\b(?:evaluat|judge|annotator|significance|confidence interval|seed|average over)\b", re.I),
    "constraint": re.compile(r"\b(?:unless otherwise|for all experiments|same setting|fair comparison|comparable|limited to|exclude|filter)\b", re.I),
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip().casefold()


def workspace_relative(path: Path, workspace_root: Path = PROJECT_ROOT) -> str:
    return path.resolve().relative_to(workspace_root.resolve()).as_posix()


def resolve_workspace_path(value: str | Path, workspace_root: Path = PROJECT_ROOT) -> Path:
    """Resolve config paths while relocating legacy absolute manifest paths."""
    workspace_root = workspace_root.resolve()
    raw = Path(value)
    if not raw.is_absolute():
        return (workspace_root / raw).resolve()

    resolved = raw.resolve()
    try:
        resolved.relative_to(workspace_root)
        return resolved
    except ValueError:
        pass

    parts = list(raw.parts)
    lower_parts = [part.casefold() for part in parts]
    root_name = workspace_root.name.casefold()
    if root_name in lower_parts:
        index = len(lower_parts) - 1 - lower_parts[::-1].index(root_name)
        candidate = workspace_root.joinpath(*parts[index + 1 :])
        if candidate.exists():
            return candidate.resolve()
    for marker in ("batch_table_text_tree", "table_context_pipeline", "external_tools"):
        if marker.casefold() in lower_parts:
            index = lower_parts.index(marker.casefold())
            return workspace_root.joinpath(*parts[index:]).resolve()
    raise ValueError(f"Absolute path is outside the workspace and cannot be relocated: {value}")


def is_reference_path(section_path: Iterable[str]) -> bool:
    return any(REFERENCE_TITLE_RE.search(str(part or "")) for part in section_path)


def stable_scope_id(section_path: Iterable[str]) -> str:
    normalized = " > ".join(normalize_text(part) for part in section_path if normalize_text(part))
    return f"scope-{sha256_text(normalized)[:16]}"


def trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _period_is_sentence_boundary(text: str, index: int, segment_start: int) -> bool:
    previous = text[index - 1] if index else ""
    following = text[index + 1] if index + 1 < len(text) else ""
    if previous.isdigit() and following.isdigit():
        return False
    prefix = text[segment_start : index + 1].casefold()
    if any(prefix.endswith(abbreviation) for abbreviation in ABBREVIATIONS):
        return False
    if re.search(r"(?:\b[a-z]\.){2,}$", prefix, re.I):
        return False
    last_word = re.search(r"([A-Za-z]+)\.$", prefix)
    if last_word and len(last_word.group(1)) == 1:
        return False
    cursor = index + 1
    while cursor < len(text) and text[cursor] in "\"'\u2019\u201d)]}":
        cursor += 1
    if cursor >= len(text):
        return True
    if text[cursor] in "\r\n":
        return True
    if not text[cursor].isspace():
        return False
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text):
        return True
    next_char = text[cursor]
    return next_char.isalpha() or next_char.isdigit() or next_char in "\"'([{\u2018\u201c"


def _sentence_ranges(text: str, start: int, end: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = start
    index = start
    while index < end:
        char = text[index]
        boundary = char in "!?\u3002\uff01\uff1f" or (
            char == "." and _period_is_sentence_boundary(text, index, cursor)
        )
        if boundary:
            split_at = index + 1
            while split_at < end and text[split_at] in "\"'\u2019\u201d)]}":
                split_at += 1
            span = trim_span(text, cursor, split_at)
            if span:
                ranges.append(span)
            cursor = split_at
            index = split_at
            continue
        index += 1
    span = trim_span(text, cursor, end)
    if span:
        ranges.append(span)
    return ranges


def _split_long_range(
    text: str, start: int, end: int, target: int = 700
) -> list[tuple[int, int]]:
    if end - start <= target:
        return [(start, end)]
    ranges: list[tuple[int, int]] = []
    cursor = start
    while end - cursor > target:
        lower = cursor + max(1, int(target * 0.65))
        upper = min(end, cursor + int(target * 1.25))
        candidates = [match.end() for match in re.finditer(r"[;,:\uff1b\uff0c]\s+", text[lower:upper])]
        if candidates:
            split_at = lower + candidates[-1]
        else:
            whitespace = [match.start() for match in re.finditer(r"\s+", text[lower:upper])]
            split_at = lower + whitespace[-1] if whitespace else min(end, cursor + target)
        span = trim_span(text, cursor, split_at)
        if span:
            ranges.append(span)
        cursor = max(split_at, cursor + 1)
    span = trim_span(text, cursor, end)
    if span:
        ranges.append(span)
    return ranges


def split_child_spans(text: str, long_span_target: int = 700) -> list[tuple[int, int]]:
    """Split on Popo paragraphs first, then abbreviation-aware sentence ends."""
    paragraph_ranges: list[tuple[int, int]] = []
    cursor = 0
    for match in POPO_SEPARATOR_RE.finditer(text or ""):
        span = trim_span(text, cursor, match.start())
        if span:
            paragraph_ranges.append(span)
        cursor = match.end()
    span = trim_span(text, cursor, len(text or ""))
    if span:
        paragraph_ranges.append(span)

    result: list[tuple[int, int]] = []
    for start, end in paragraph_ranges:
        for sentence_start, sentence_end in _sentence_ranges(text, start, end):
            result.extend(
                _split_long_range(
                    text,
                    sentence_start,
                    sentence_end,
                    target=long_span_target,
                )
            )
    return result


def make_children(parent_id: str, full_text: str) -> list[dict[str, Any]]:
    children = []
    for index, (start, end) in enumerate(split_child_spans(full_text), start=1):
        child_text = full_text[start:end]
        children.append(
            {
                "child_id": f"{parent_id}-child-{index:04d}",
                "parent_id": parent_id,
                "char_start": start,
                "char_end": end,
                "text": child_text,
                "text_sha256": sha256_text(child_text),
            }
        )
    return children


def _normalize_designator(value: str) -> str:
    value = re.sub(r"\s+", "", value or "")
    value = re.sub(r"\([^)]*\)$", "", value)
    return value.casefold()


def _label_designator(label: str) -> str:
    return _normalize_designator(
        re.sub(r"^\s*(?:table|tab|tbl)\.?\s*", "", label or "", flags=re.I)
    )


def _numeric_designator(value: str) -> tuple[str, int, str] | None:
    match = re.fullmatch(r"([A-Za-z]?)(\d+)([A-Za-z]?)", _normalize_designator(value))
    if not match:
        return None
    return match.group(1), int(match.group(2)), match.group(3)


def parse_table_references(
    text: str, tables: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    designator_to_ids: dict[str, list[str]] = {}
    for table in tables:
        designator = _label_designator(table.get("canonical_label") or "")
        if designator:
            designator_to_ids.setdefault(designator, []).append(table["table_id"])

    mentions = []
    for prefix in TABLE_PREFIX_RE.finditer(text or ""):
        first = DESIGNATOR_RE.match(text, prefix.end())
        if not first:
            continue
        tokens = [first.group(0)]
        connectors: list[str] = []
        cursor = first.end()
        while True:
            connector = CONNECTOR_RE.match(text, cursor)
            if not connector:
                break
            designator = DESIGNATOR_RE.match(text, connector.end())
            if not designator:
                break
            connectors.append(connector.group("connector").strip())
            tokens.append(designator.group(0))
            cursor = designator.end()

        normalized_tokens = [_normalize_designator(token) for token in tokens]
        expanded = [normalized_tokens[0]]
        for left, connector, right in zip(tokens, connectors, tokens[1:]):
            if connector.casefold() not in {"-", "\u2013", "\u2014", "to"}:
                expanded.append(_normalize_designator(right))
                continue
            left_number = _numeric_designator(left)
            right_number = _numeric_designator(right)
            if not left_number or not right_number:
                expanded.append(_normalize_designator(right))
                continue
            if left_number[0] != right_number[0] or left_number[2] or right_number[2]:
                expanded.append(_normalize_designator(right))
                continue
            low, high = left_number[1], right_number[1]
            if 0 <= high - low <= 100:
                expanded.extend(
                    f"{left_number[0]}{value}".casefold()
                    for value in range(low + 1, high + 1)
                )
            else:
                expanded.append(_normalize_designator(right))

        table_ids = []
        for designator in expanded:
            for table_id in designator_to_ids.get(designator, []):
                if table_id not in table_ids:
                    table_ids.append(table_id)
        mentions.append(
            {
                "char_start": prefix.start(),
                "char_end": cursor,
                "text": text[prefix.start() : cursor],
                "designators": normalized_tokens,
                "resolved_table_ids": table_ids,
            }
        )
    return mentions


def condition_hints(text: str) -> list[str]:
    return sorted(name for name, pattern in CONDITION_HINT_PATTERNS.items() if pattern.search(text or ""))


def lexical_tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in WORD_RE.findall(unicodedata.normalize("NFKC", text or ""))
        if len(token) > 2
    }


def paths_share_section(left: Iterable[str], right: Iterable[str]) -> bool:
    left_parts = [normalize_text(part) for part in left if normalize_text(part)]
    right_parts = [normalize_text(part) for part in right if normalize_text(part)]
    if not left_parts or not right_parts:
        return False
    common = 0
    for left_part, right_part in zip(left_parts, right_parts):
        if left_part != right_part:
            break
        common += 1
    return common == min(len(left_parts), len(right_parts))


def minimum_distance(left: Iterable[int], right: Iterable[int]) -> int | None:
    left_values = list(left)
    right_values = list(right)
    if not left_values or not right_values:
        return None
    return min(abs(a - b) for a in left_values for b in right_values)


def build_table_suggestions(
    child: dict[str, Any], parent: dict[str, Any], tables: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    mentions = parse_table_references(child["text"], tables)
    explicitly_referenced = {
        table_id for mention in mentions for table_id in mention["resolved_table_ids"]
    }
    hints = condition_hints(child["text"])
    child_tokens = lexical_tokens(child["text"])
    suggestions = []
    for table in tables:
        channels = []
        score_components: dict[str, int] = {}
        if table["table_id"] in explicitly_referenced:
            channels.append("structured_table_reference")
            score_components["structured_table_reference"] = 1000
        if paths_share_section(parent.get("section_path") or [], table.get("section_path") or []):
            channels.append("same_section")
            score_components["same_section"] = 200
        page_distance = minimum_distance(
            parent.get("page_indices") or [], table.get("page_indices") or []
        )
        if page_distance is not None and page_distance <= 2:
            channels.append("nearby_page")
            score_components["nearby_page"] = 100 if page_distance == 0 else 60 if page_distance == 1 else 30
        block_distance = minimum_distance(
            parent.get("block_ids") or [], table.get("popo_block_ids") or []
        )
        if block_distance is not None and block_distance <= 20:
            channels.append("nearby_reading_order")
            score_components["nearby_reading_order"] = 80 if block_distance <= 5 else 40 if block_distance <= 10 else 20
        if hints:
            channels.append("condition_hints")
            score_components["condition_hints"] = min(90, len(hints) * 10)
        overlap = sorted(child_tokens & set(table.get("lexical_tokens") or []))
        if overlap:
            channels.append("table_text_lexical_overlap")
            score_components["table_text_lexical_overlap"] = min(50, len(overlap) * 5)
        if not channels:
            continue
        suggestions.append(
            {
                "table_id": table["table_id"],
                "channels": channels,
                "ordering_score": sum(score_components.values()),
                "score_components": score_components,
                "page_distance": page_distance,
                "block_distance": block_distance,
                "condition_hints": hints,
                "lexical_overlap_sample": overlap[:20],
                "binding": False,
            }
        )
    suggestions.sort(key=lambda row: (-row["ordering_score"], row["table_id"]))
    return suggestions, mentions, hints


def apply_anchor_overrides(
    slug: str,
    anchors: list[dict[str, Any]],
    overrides: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    updated = copy.deepcopy(anchors)
    audit = {"status": "pass", "applied": [], "failures": []}
    relevant = [row for row in overrides if row.get("slug") == slug]
    seen: set[str] = set()
    for override in relevant:
        label = str(override.get("canonical_label") or "")
        normalized_label = normalize_text(label)
        if normalized_label in seen:
            audit["failures"].append(
                {"reason": "duplicate_anchor_override", "canonical_label": label}
            )
            continue
        seen.add(normalized_label)
        matches = [
            (index, anchor)
            for index, anchor in enumerate(updated)
            if normalize_text(anchor.get("canonical_label") or "") == normalized_label
        ]
        if len(matches) != 1:
            audit["failures"].append(
                {
                    "reason": "anchor_override_target_count",
                    "canonical_label": label,
                    "target_count": len(matches),
                }
            )
            continue
        index, anchor = matches[0]
        before = {
            "page_index": anchor.get("page_index"),
            "bbox": anchor.get("bbox"),
            "caption": anchor.get("caption"),
        }
        for field in ("page_index", "bbox", "caption"):
            if field in override:
                updated[index][field] = copy.deepcopy(override[field])
        audit["applied"].append(
            {
                "canonical_label": label,
                "before": before,
                "after": {
                    "page_index": updated[index].get("page_index"),
                    "bbox": updated[index].get("bbox"),
                    "caption": updated[index].get("caption"),
                },
                "reason": override.get("reason"),
            }
        )
    if audit["failures"]:
        audit["status"] = "fail"
    return updated, audit
