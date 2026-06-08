import argparse
import json
import logging
import re
import time
from html.parser import HTMLParser
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_OCR_FILE = Path.cwd() / "data" / "full_results.json"
DEFAULT_FALLBACK_OCR_FILE = None
DEFAULT_OUTPUT_DIR = Path.cwd() / "outputs" / "ocr_glm"


def normalize_space(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def remove_control_chars(text):
    return re.sub(r"[\x00-\x1f]+", " ", str(text or ""))


def strip_code_fences(text):
    text = str(text or "").strip()
    text = re.sub(r"^\s*```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def extract_json(text):
    if not text:
        return "{}"
    text = remove_control_chars(strip_code_fences(text))
    start_idx = text.find("{")
    if start_idx == -1:
        return text

    depth = 0
    in_string = False
    escape = False
    for idx in range(start_idx, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx : idx + 1]

    end_idx = text.rfind("}")
    if end_idx > start_idx:
        logger.warning("Possible output truncation: JSON object did not close cleanly during extraction.")
        return text[start_idx : end_idx + 1]
    logger.warning("Possible output truncation: no closing JSON brace found.")
    return text


def robust_json_loads(text, context=""):
    cleaned = strip_code_fences(text)
    candidates = [cleaned, remove_control_chars(cleaned)]
    extracted = extract_json(cleaned)
    if extracted not in candidates:
        candidates.append(extracted)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            pass
        try:
            obj, _ = decoder.raw_decode(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            fixed = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", candidate)
            return json.loads(fixed)
        except Exception:
            pass

    try:
        import ast

        py_text = candidates[-1].replace("true", "True").replace("false", "False").replace("null", "None")
        obj = ast.literal_eval(py_text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    logger.warning("JSON parse failed in %s", context)
    return {}


def load_json_list(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON list")
    return data


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def split_citekey_evidences(cited_facts_text):
    text = cited_facts_text or ""
    pattern = re.compile(r"citekey evidence\s+(\d+)\s*\((.*?)\):\s*", flags=re.I | re.S)
    matches = list(pattern.finditer(text))
    evidences = []
    if not matches:
        compact = normalize_space(text)
        return [{"index": 1, "source": "", "text": compact}] if compact else []
    for pos, match in enumerate(matches):
        start = match.end()
        end = matches[pos + 1].start() if pos + 1 < len(matches) else len(text)
        content = normalize_space(text[start:end])
        if content:
            evidences.append(
                {
                    "index": int(match.group(1)),
                    "source": normalize_space(match.group(2)),
                    "text": content,
                }
            )
    return evidences


class TableHTMLParser(HTMLParser):
    """Small table parser for Paddle/PP-Structure HTML table snippets."""

    def __init__(self):
        super().__init__()
        self.rows = []
        self.in_thead = False
        self.in_tbody = False
        self.in_tr = False
        self.in_cell = False
        self.current_row = None
        self.current_cell = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "thead":
            self.in_thead = True
        elif tag == "tbody":
            self.in_tbody = True
        elif tag == "tr":
            self.in_tr = True
            section = "thead" if self.in_thead else "tbody" if self.in_tbody else "body"
            self.current_row = {"section": section, "cells": []}
        elif tag in {"td", "th"} and self.in_tr:
            self.in_cell = True
            self.current_cell = {
                "tag": tag,
                "text_parts": [],
                "rowspan": int(attrs.get("rowspan", "1") or 1),
                "colspan": int(attrs.get("colspan", "1") or 1),
            }

    def handle_data(self, data):
        if self.in_cell and self.current_cell is not None:
            text = normalize_space(data)
            if text:
                self.current_cell["text_parts"].append(text)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.current_cell is not None:
            cell = {
                "tag": self.current_cell["tag"],
                "text": normalize_space(" ".join(self.current_cell["text_parts"])),
                "rowspan": max(1, self.current_cell["rowspan"]),
                "colspan": max(1, self.current_cell["colspan"]),
            }
            self.current_row["cells"].append(cell)
            self.current_cell = None
            self.in_cell = False
        elif tag == "tr" and self.current_row is not None:
            self.rows.append(self.current_row)
            self.current_row = None
            self.in_tr = False
        elif tag == "thead":
            self.in_thead = False
        elif tag == "tbody":
            self.in_tbody = False


def parse_html_table(html):
    html = html or ""
    if not html.strip():
        return []
    parser = TableHTMLParser()
    parser.feed(html)
    return parser.rows


def expand_rows(raw_rows):
    grid_rows = []
    spans = {}

    for row in raw_rows:
        grid = []
        col = 0

        def fill_span_at(column):
            span = spans.get(column)
            if not span:
                return False
            while len(grid) <= column:
                grid.append("")
            grid[column] = span["text"]
            span["remaining"] -= 1
            if span["remaining"] <= 0:
                del spans[column]
            return True

        for cell in row.get("cells", []):
            while fill_span_at(col):
                col += 1
            text = normalize_space(cell.get("text", ""))
            colspan = max(1, int(cell.get("colspan", 1) or 1))
            rowspan = max(1, int(cell.get("rowspan", 1) or 1))
            for offset in range(colspan):
                while len(grid) <= col + offset:
                    grid.append("")
                grid[col + offset] = text
                if rowspan > 1:
                    spans[col + offset] = {"text": text, "remaining": rowspan - 1}
            col += colspan

        while spans:
            max_span_col = max(spans)
            if col > max_span_col:
                break
            fill_span_at(col)
            col += 1

        grid_rows.append(
            {
                "section": row.get("section", "body"),
                "cells": [normalize_space(cell) for cell in grid],
            }
        )

    return grid_rows


def header_path(header_rows, col_index):
    parts = []
    for row in header_rows:
        cells = row.get("cells", [])
        if col_index < len(cells):
            text = normalize_space(cells[col_index])
            if text and (not parts or parts[-1] != text):
                parts.append(text)
    return " / ".join(parts)


def table_rows_to_text(expanded_rows):
    if not expanded_rows:
        return "", []

    header_rows = [row for row in expanded_rows if row.get("section") == "thead"]
    body_rows = [row for row in expanded_rows if row.get("section") != "thead"]
    if not header_rows and expanded_rows:
        header_rows = expanded_rows[:1]
        body_rows = expanded_rows[1:]

    max_cols = max((len(row.get("cells", [])) for row in expanded_rows), default=0)
    headers = [header_path(header_rows, i) or f"Column {i + 1}" for i in range(max_cols)]

    section_context = []
    lines = []
    structured_rows = []

    for row_index, row in enumerate(body_rows, 1):
        cells = row.get("cells", [])
        non_empty = [cell for cell in cells if normalize_space(cell)]
        if len(non_empty) == 1 and len(cells) > 1:
            section_context = [non_empty[0]]
            lines.append(f"Section: {non_empty[0]}")
            structured_rows.append(
                {
                    "row_index": row_index,
                    "row_type": "section",
                    "section_context": section_context[:],
                    "row_text": f"Section: {non_empty[0]}",
                    "cells": cells,
                }
            )
            continue

        pairs = []
        for col_index in range(max_cols):
            value = normalize_space(cells[col_index] if col_index < len(cells) else "")
            if not value:
                continue
            header = headers[col_index] if col_index < len(headers) else f"Column {col_index + 1}"
            pairs.append(f"{header}: {value}")

        if not pairs:
            continue

        prefix = f"Row {row_index}"
        if section_context:
            prefix += f" | Section context: {' > '.join(section_context)}"
        row_text = prefix + " | " + " ; ".join(pairs)
        lines.append(row_text)
        structured_rows.append(
            {
                "row_index": row_index,
                "row_type": "data",
                "section_context": section_context[:],
                "row_text": row_text,
                "cells": cells,
            }
        )

    return "\n".join(lines), structured_rows


def compact_author_key(text):
    text = str(text or "").lower()
    text = re.sub(r"et\s*al\.?", "etal", text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def author_tokens(author):
    author = str(author or "")
    years = re.findall(r"\b\d{4}[a-z]?\b", author, flags=re.I)
    cleaned = re.sub(r"\b\d{4}[a-z]?\b", " ", author, flags=re.I)
    cleaned = re.sub(r"et\s*al\.?|etal\.?|and|&", " ", cleaned, flags=re.I)
    tokens = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z'-]+", cleaned)]
    surname = tokens[0] if tokens else ""
    return surname, years


def find_candidate_rows(structured_rows, author):
    compact_author = compact_author_key(author)
    surname, years = author_tokens(author)
    candidates = []
    for row in structured_rows:
        row_text = row.get("row_text", "")
        compact_row = compact_author_key(row_text)
        score = 0
        reasons = []
        if compact_author and compact_author in compact_row:
            score += 5
            reasons.append("compact_author")
        if surname and surname in compact_row:
            score += 2
            reasons.append("surname")
        for year in years:
            if year.lower() in compact_row:
                score += 2
                reasons.append(f"year:{year}")
        if score >= 3:
            candidate = dict(row)
            candidate["match_score"] = score
            candidate["match_reason"] = reasons
            candidates.append(candidate)
    candidates.sort(key=lambda item: item.get("match_score", 0), reverse=True)
    return candidates


def get_first_original_analysis(item):
    analyses = item.get("originalkey_analysis") or []
    if not analyses:
        return None
    return analyses[0]


def get_raw_structure(analysis):
    payload = analysis.get("analysis") or {}
    table_structure = payload.get("table_structure_analysis") or {}
    raw_results = table_structure.get("raw_structure_result") or []
    return raw_results[0] if raw_results else {}


def get_pred_html(raw_structure):
    for table_result in raw_structure.get("table_res_list") or []:
        pred_html = table_result.get("pred_html")
        if normalize_space(pred_html):
            return pred_html
    parsing = raw_structure.get("parsing_res_list") or []
    for text in parsing:
        match = re.search(r"(<html>.*?</html>)", str(text), flags=re.S | re.I)
        if match:
            return match.group(1)
    return ""


def get_ocr_texts(raw_structure):
    texts = []
    overall = raw_structure.get("overall_ocr_res") or {}
    texts.extend(overall.get("rec_texts") or [])
    for table_result in raw_structure.get("table_res_list") or []:
        table_ocr = table_result.get("table_ocr_pred") or {}
        texts.extend(table_ocr.get("rec_texts") or [])
    deduped = []
    seen = set()
    for text in texts:
        text = normalize_space(text)
        if text and text not in seen:
            seen.add(text)
            deduped.append(text)
    return deduped


def truncate_text(text, max_chars):
    text = text or ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    logger.warning("Input truncation: text length %s exceeds max_chars=%s.", len(text), max_chars)
    return text[:max_chars] + "\n...[TRUNCATED]"


def join_citekey_facts(citekey_analysis, max_chars):
    facts = []
    if isinstance(citekey_analysis, dict):
        iterable = citekey_analysis.items()
    elif isinstance(citekey_analysis, list):
        iterable = enumerate(citekey_analysis)
    else:
        iterable = []
    for index, value in iterable:
        if not isinstance(value, dict):
            continue
        description = normalize_space(value.get("description"))
        if description:
            facts.append(f"citekey evidence {len(facts) + 1} ({index}): {description}")
    return truncate_text("\n\n".join(facts), max_chars)


def build_bundle(item, fallback_by_id=None, max_table_chars=0, max_ocr_texts=0, max_citekey_chars=0):
    fallback_by_id = fallback_by_id or {}
    original = get_first_original_analysis(item)
    fallback_used = False

    if original is None:
        fallback = fallback_by_id.get(item.get("id"))
        original = get_first_original_analysis(fallback or {})
        fallback_used = original is not None

    payload = (original or {}).get("analysis") or {}
    raw_structure = get_raw_structure(original or {})
    pred_html = get_pred_html(raw_structure)
    expanded_rows = expand_rows(parse_html_table(pred_html))
    table_rows_text, structured_rows = table_rows_to_text(expanded_rows)
    ocr_texts = get_ocr_texts(raw_structure)
    candidates = find_candidate_rows(structured_rows, item.get("author"))

    if not table_rows_text and payload.get("description"):
        table_rows_text = payload.get("description", "")

    tab_id = (original or {}).get("tab_id", "")
    caption = normalize_space(payload.get("caption") or payload.get("caption_annotations"))
    cited_facts_text = join_citekey_facts(item.get("citekey_analysis"), max_citekey_chars)

    candidate_text = "\n".join(
        f"Candidate {idx + 1}: {row.get('row_text')} | reason={','.join(row.get('match_reason', []))}"
        for idx, row in enumerate(candidates[:8])
    )

    if max_ocr_texts > 0 and len(ocr_texts) > max_ocr_texts:
        logger.warning(
            "Input truncation: OCR text count %s exceeds max_ocr_texts=%s for sample %s.",
            len(ocr_texts),
            max_ocr_texts,
            item.get("id") or item.get("paper_id"),
        )
        raw_ocr_values = ocr_texts[:max_ocr_texts]
    else:
        raw_ocr_values = ocr_texts
    raw_ocr_text = " | ".join(raw_ocr_values)
    table_rows_text = truncate_text(table_rows_text, max_table_chars)

    return {
        "id": item.get("id") or item.get("paper_id"),
        "paper_id": item.get("paper_id") or item.get("id"),
        "citekey": item.get("citekey"),
        "originalkey": item.get("originalkey"),
        "author": item.get("author"),
        "label": item.get("label"),
        "charts": item.get("charts") or [],
        "tab_id": tab_id,
        "caption": caption,
        "cited_facts_text": cited_facts_text,
        "target_claims_text": candidate_text,
        "table_rows_text": table_rows_text,
        "candidate_rows_text": candidate_text,
        "raw_ocr_text": raw_ocr_text,
        "source_quality": {
            "processing_status": item.get("processing_status"),
            "used_fallback_originalkey_analysis": fallback_used,
            "has_pred_html": bool(pred_html),
            "num_structured_rows": len(structured_rows),
            "num_candidate_rows": len(candidates),
            "num_ocr_texts": len(ocr_texts),
            "existing_author_related_content_nonempty": bool(
                normalize_space(payload.get("author_related_content"))
            ),
            "error_message": item.get("error_message") or "",
        },
    }


def build_extraction_prompt(bundle):
    return f"""【任务目标】
根据 OCR 表格结果，重新提取与目标引用作者相关的 originalkey/施引侧表格内容，用于后续事实核查。

【目标引用作者】
{bundle['author']}

【表格编号】
{bundle['tab_id'] or '未知'}

【表格 caption】
{bundle['caption'] or '无'}

【程序预匹配的候选行】
{bundle['candidate_rows_text'] or '未找到候选行，请你根据完整 OCR 表格自行定位。'}

【完整 OCR 表格行】
{bundle['table_rows_text'] or '无结构化表格行'}

【OCR 原始识别文本，作为补充】
{bundle['raw_ocr_text'] or '无'}

【关键规则】
1. 目标引用作者只作为定位信号，例如 "(Wang et al., 2023)"、"Gellaetal.,2017"、"[12]" 等。
2. 请先定位该引用标识所在的行、列或候选区域，但不要只摘取表面上一整行的文字。
3. 对每个数值都要尽量判断其完整单元格归属：分组行/section、行头、列头、多级表头、caption 中的指标解释，以及该数值本身。
4. 输出的 author_related_content 应删除纯引用标识本身，但保留模型名、方法名、数据集、指标、数值和必要的表格上下文。
5. 可以使用一般学术常识辅助理解指标、缩写和任务背景，但不得补全 OCR 中缺失的事实证据。
6. 若 OCR 表格有多级表头，请把多级表头合并到指标名中，例如 "Image to Text / R@1: 31.7"。
7. 若 OCR 把多个作者、多个指标或多个数值合并到同一个单元格，请保留歧义并说明，不要强行拆分。
8. 若存在多个可能匹配行或列，请全部列出并说明原因；不要强行只选一个。

【输出格式】
请只返回严格 JSON，不要返回 Markdown。
{{
  "target_found": true,
  "target_author": "{bundle['author']}",
  "tab_id": "{bundle['tab_id']}",
  "caption_annotations": "从 caption 中抽取的表格说明，如指标含义、数据集、缩写解释；没有则为空字符串",
  "author_related_content": "与目标引用作者相关的表格内容，删除纯引用标识后保留方法/模型/指标/数值",
  "evidence_rows": [
    {{
      "row_or_column": "定位到的行或列的简短说明",
      "content": "该行或列的完整结构化内容",
      "match_reason": "为什么认为它对应目标引用作者"
    }}
  ],
  "cell_attributions": [
    {{
      "value": "数值或关键单元格文本",
      "row_context": "该值所属的行头/模型/方法/作者上下文",
      "column_context": "该值所属的列头/多级表头/指标上下文",
      "section_context": "该值所属的分组、数据集、实验设置等上下文；没有则为空字符串",
      "caption_context": "caption 中解释该值或指标的相关内容；没有则为空字符串",
      "attribution_note": "该归属是否确定，若 OCR 合并或错位请说明"
    }}
  ],
  "extraction_status": "matched / ambiguous / not_found",
  "notes": "简短说明 OCR 噪声、合并单元格、多候选等情况"
}}
"""


def build_pairwise_comparison_prompt(row, citekey_evidence):
    prompt_inputs = row.get("prompt_inputs") or {}
    author = row.get("author") or ""
    tab_id = prompt_inputs.get("tab_id", "")
    caption = prompt_inputs.get("caption", "")
    target_claims = prompt_inputs.get("target_claims_text") or prompt_inputs.get("candidate_rows_text", "")
    evidence_label = f"citekey evidence {citekey_evidence.get('index')}"
    if citekey_evidence.get("source"):
        evidence_label += f" ({citekey_evidence.get('source')})"

    return f"""你是一名严谨的学术事实核查员（Academic Fact-Checker）。

【任务目标】
验证【待核查的引用句】中的事实陈述，是否准确反映了【原文事实】中的数据。

【样本信息】
sample_id: {row.get('id') or row.get('paper_id') or ''}
target_author: {author}
originalkey_tab_id: {tab_id or '未知'}

【输入数据】
<原文事实 (Ground Truth)：当前 citekey OCR evidence>
{evidence_label}
{citekey_evidence.get('text') or '无'}
</原文事实 (Ground Truth)>

<待核查的引用句 (Citation Sentence)：originalkey target claims>
目标引用作者 / 定位信号：{author}
{target_claims or '未找到明确 target claims；若无法确定待核查事实，target_found=false 且 score 应较低。'}
</待核查的引用句 (Citation Sentence)>

【表格注释信息】
{caption or '无表格注释信息'}

【关键定义：引用标识】
在"待核查的引用句"中，可能会出现 {author} 这样的引用标识，类似 `(Author, Year)` 或 `[1]`，括号内容视为**"引用标识"**。
1. **不作为核查对象**：不要把引用标识本身当作需要和"原文事实"逐字匹配的事实。
2. **作为定位信号**：引用标识已用于抽取待核查的 originalkey target claims。
3. **核查重点**：删除引用标识本身，重点核查其管辖的文本内容（数值、结论、模型表现、数据集、指标归属）是否与"原文事实"一致。

【核心核查原则】
1. **事实支持原则**：
   - "待核查的引用句"中被引用标识管辖的陈述，必须在"原文事实"中找到确凿证据。
   - 允许引用句只提取原文的部分信息（如原文有 R-1/R-2/R-L，引用句只提 R-L），这属于**合理概括**，不扣分。
   - 如果"原文事实"无法提供可核查的表格事实，或虽然是表格但与待核查内容无关，应判为 score=0.0。
2. **严禁张冠李戴**：检查数值是否归属于正确的评估指标、模型、方法、数据集和实验设置。
3. **数值与逻辑一致性**：数值允许合理的四舍五入误差。
4. **表格注释参考**：结合注释信息理解表格内容。
5. **OCR 归属参考**：如果 OCR 合并单元格、错位或多候选导致不确定，请结合行头、列头、多级表头和 caption 说明归属。

【评分规则】
- **Score = 1.0**: 引用句中的所有关键事实均准确无误，且归属正确。
- **0.5 < Score < 1.0**: 事实基本准确，存在非关键信息的模糊表述、轻微 OCR 噪声或四舍五入差异。
- **Score <= 0.5**: 存在数值错误、模型归属错误、指标/数据集归属错误、缺少关键事实支持或捏造结论。
- **Score = 0.0**: 原文事实无法提供可核查的表格事实，或与待核查引用句无关。
- 程序将在模型返回后根据 score > 0.5 判定 match；模型只需准确选择 score。

【输出格式】
请仅返回严格的 JSON 格式对象，不要包含 Markdown 标记。score 写 0.0~1.0 范围内的数字。
{{
  "citekey_evidence_index": {citekey_evidence.get('index', 1)},
  "citekey_evidence_source": "{str(citekey_evidence.get('source', '')).replace('"', "'")}",
  "originalkey_evidence": {{
    "target_found": true/false,
    "author_related_content": "从待核查引用句中整理出的目标作者相关内容",
    "caption_annotations": "caption 中对指标、缩写、数据集或实验设置的说明；没有则为空字符串",
    "cell_attributions": [
      {{
        "value": "数值或关键单元格文本",
        "row_context": "该值所属的行头/模型/方法/作者上下文",
        "column_context": "该值所属的列头/多级表头/指标上下文",
        "section_context": "该值所属的分组、数据集、实验设置等上下文；没有则为空字符串",
        "caption_context": "caption 中解释该值或指标的相关内容；没有则为空字符串",
        "attribution_note": "该归属是否确定，若 OCR 合并或错位请说明"
      }}
    ]
  }},
  "score": 0.0~1.0,
  "explanation": "简短说明本条 citekey evidence 与 originalkey target claims 一致或不一致的依据",
  "extraction_status": "matched / ambiguous / not_found"
}}
"""


def build_system_prompt(prompt_kind):
    if prompt_kind == "pairwise":
        return (
            "你是一名严谨的学术 OCR 表格事实核查员。"
            "你的任务是基于 OCR 结果判断 originalkey 表格中由目标引用标识管辖的具体 claims，"
            "是否被当前这一条 citekey OCR evidence 支持。"
            "citekey OCR evidence 是被引侧事实来源；originalkey OCR table evidence 是施引侧待核查证据，"
            "不能把 originalkey 的表格内容同时当作待核查事实和事实真值。"
            "不能因为主题相似、同一论文其他图片可能支持、或完整 original 表格中存在相似文本就给高分。"
            "不得自行补全缺失信息，输出必须是严格 JSON。"
        )
    return (
        "你是一名严谨的学术表格 OCR 信息抽取助手。"
        "你的任务是从 OCR 表格结构中重新抽取与目标引用作者相关的 originalkey 表格证据。"
        "你必须尽量保留每个数值的单元格归属，包括分组行、行头、列头、多级表头和 caption 解释。"
        "输出必须是严格 JSON。"
    )


def parse_model_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1", "match", "matched"}:
            return True
        if lowered in {"false", "no", "n", "0", "mismatch", "not_match", "not matched", "unmatched"}:
            return False
    return False


def normalize_match_score(parsed, context="", match_threshold=0.5):
    if not isinstance(parsed, dict):
        return False, 0.0
    raw_score = parsed.get("score", None)
    try:
        score = float(raw_score)
    except Exception:
        score = 0.0
        logger.warning("Score missing or invalid in %s; use score=0.0 for aggregation.", context)
    score = max(0.0, min(1.0, score))
    match = score > match_threshold
    return match, score


def call_glm_text_model(system_prompt, user_prompt, api_base, api_key, model, max_tokens, temperature):
    from openai import OpenAI

    if "[TRUNCATED]" in (system_prompt or "") or "[TRUNCATED]" in (user_prompt or ""):
        logger.warning("Input truncation marker detected in prompt sent to model.")

    client = OpenAI(base_url=api_base, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length":
        logger.warning("Possible output truncation: model response finish_reason=length.")
    content = choice.message.content or ""
    return content.strip()


def convert_model_output_to_originalkey_analysis(bundle, parsed):
    if "originalkey_evidence" in parsed:
        evidence = parsed.get("originalkey_evidence") or {}
        author_content = evidence.get("author_related_content", "")
        caption_annotations = evidence.get("caption_annotations", "")
        cell_attributions = evidence.get("cell_attributions", [])
        status = parsed.get("extraction_status", "")
    else:
        author_content = parsed.get("author_related_content", "")
        caption_annotations = parsed.get("caption_annotations", "")
        cell_attributions = parsed.get("cell_attributions", [])
        status = parsed.get("extraction_status", "")

    description_parts = []
    if bundle.get("table_rows_text"):
        description_parts.append(bundle["table_rows_text"])
    if bundle.get("caption"):
        description_parts.append(f"Caption: {bundle['caption']}")

    return {
        "tab_id": bundle.get("tab_id", ""),
        "analysis": {
            "description": "\n".join(description_parts),
            "author_related_content": normalize_space(author_content),
            "caption_annotations": normalize_space(caption_annotations or bundle.get("caption", "")),
            "caption": bundle.get("caption", ""),
            "ocr_extraction_status": status,
            "cell_attributions": cell_attributions if isinstance(cell_attributions, list) else [],
            "source_quality": bundle.get("source_quality", {}),
        },
    }


def get_originalkey_evidence_from_parsed(parsed):
    if not isinstance(parsed, dict):
        parsed = {}
    evidence = parsed.get("originalkey_evidence")
    if not isinstance(evidence, dict):
        evidence = {
            "target_found": parsed.get("target_found"),
            "author_related_content": parsed.get("author_related_content", ""),
            "caption_annotations": parsed.get("caption_annotations", ""),
            "cell_attributions": parsed.get("cell_attributions", []),
        }
    cell_attributions = evidence.get("cell_attributions", [])
    if not isinstance(cell_attributions, list):
        cell_attributions = []
    return {
        "target_found": parse_model_bool(evidence.get("target_found", False)),
        "author_related_content": normalize_space(evidence.get("author_related_content", "")),
        "caption_annotations": normalize_space(evidence.get("caption_annotations", "")),
        "cell_attributions": cell_attributions,
    }


def build_clean_pairwise_comparison(comparison):
    evidence = comparison.get("citekey_evidence") or {}
    parsed = comparison.get("parsed_model_result") or {}
    originalkey_evidence = get_originalkey_evidence_from_parsed(parsed)
    return {
        "citekey_evidence_index": evidence.get("index"),
        "citekey_evidence_source": evidence.get("source", ""),
        "citekey_evidence_text": evidence.get("text", ""),
        "score": comparison.get("score", 0.0),
        "match": comparison.get("match", False),
        "explanation": normalize_space(parsed.get("explanation", "")),
        "extraction_status": normalize_space(parsed.get("extraction_status", "")),
        "author_related_content": originalkey_evidence["author_related_content"],
        "originalkey_evidence": originalkey_evidence,
    }


def sort_comparisons_for_review(comparisons):
    return sorted(
        comparisons,
        key=lambda item: (
            -float(item.get("score", 0.0) or 0.0),
            (item.get("citekey_evidence") or {}).get("index") or 0,
        ),
    )


def build_full_pairwise_output(row, comparisons, sample_failures, match_threshold):
    sorted_comparisons = sort_comparisons_for_review(comparisons)
    best = sorted_comparisons[0] if sorted_comparisons else None
    final_score = float(best.get("score", 0.0)) if best else 0.0
    final_match = final_score > match_threshold

    output = dict(row)
    output["task"] = "pairwise_citekey_originalkey_ocr_consistency"
    output["citekey_comparisons"] = sorted_comparisons
    output["num_citekey_evidences"] = len(sorted_comparisons)
    output["final_score"] = final_score
    output["final_match"] = final_match
    output["result"] = "true" if final_match else "false"
    output["best_citekey_evidence_index"] = (
        (best.get("citekey_evidence") or {}).get("index") if best else None
    )
    output["sample_failures"] = sample_failures
    return output


def build_clean_pairwise_output(row, comparisons, sample_failures, match_threshold):
    prompt_inputs = row.get("prompt_inputs") or {}
    clean_comparisons = [
        build_clean_pairwise_comparison(item)
        for item in sort_comparisons_for_review(comparisons)
    ]
    for rank, item in enumerate(clean_comparisons, 1):
        item["rank"] = rank

    best = clean_comparisons[0] if clean_comparisons else None
    final_score = float(best.get("score", 0.0)) if best else 0.0
    final_match = final_score > match_threshold

    return {
        "id": row.get("id"),
        "paper_id": row.get("paper_id"),
        "citekey": row.get("citekey"),
        "originalkey": row.get("originalkey"),
        "author": row.get("author"),
        "label": row.get("label"),
        "task": "pairwise_citekey_originalkey_ocr_consistency",
        "originalkey_context": {
            "tab_id": prompt_inputs.get("tab_id", ""),
            "caption": prompt_inputs.get("caption", ""),
            "target_claims_text": prompt_inputs.get("target_claims_text", ""),
            "table_rows_text": prompt_inputs.get("table_rows_text", ""),
            "source_quality": prompt_inputs.get("source_quality", {}),
        },
        "num_citekey_evidences": len(clean_comparisons),
        "final_score": final_score,
        "final_match": final_match,
        "result": "true" if final_match else "false",
        "best_citekey_evidence_index": best.get("citekey_evidence_index") if best else None,
        "best_author_related_content": best.get("author_related_content", "") if best else "",
        "citekey_comparisons": clean_comparisons,
        "sample_failures": sample_failures,
    }


def build_output_record(item, bundle, system_prompt, user_prompt, prompt_kind, model_result=None, parsed=None):
    output = {
        "id": bundle["id"],
        "paper_id": bundle["paper_id"],
        "citekey": bundle["citekey"],
        "originalkey": bundle["originalkey"],
        "author": bundle["author"],
        "label": bundle["label"],
        "charts": bundle["charts"],
        "prompt_kind": prompt_kind,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "prompt": user_prompt,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "prompt_inputs": {
            "tab_id": bundle["tab_id"],
            "caption": bundle["caption"],
            "cited_facts_text": bundle["cited_facts_text"],
            "target_claims_text": bundle["target_claims_text"],
            "candidate_rows_text": bundle["candidate_rows_text"],
            "table_rows_text": bundle["table_rows_text"],
            "raw_ocr_text": bundle["raw_ocr_text"],
            "source_quality": bundle["source_quality"],
        },
        "processing_status": item.get("processing_status"),
        "error_message": item.get("error_message") or "",
    }

    if model_result is not None:
        output["model_result"] = model_result
        output["parsed_model_result"] = parsed or {}
        output["originalkey_analysis"] = [
            convert_model_output_to_originalkey_analysis(bundle, parsed or {})
        ]

    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-integrate OCR full_results into GLM4.6V-flash text prompts/results."
    )
    parser.add_argument("--ocr-file", type=Path, default=DEFAULT_OCR_FILE)
    parser.add_argument("--fallback-ocr-file", type=Path, default=DEFAULT_FALLBACK_OCR_FILE)
    parser.add_argument(
        "--input-prompts",
        type=Path,
        default=None,
        help="Run model from an existing prompt JSONL. Use this on the remote server.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-kind", choices=["extraction", "pairwise"], default="extraction")
    parser.add_argument("--run-model", action="store_true", help="Call local vLLM/OpenAI-compatible GLM server.")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="your-api-key")
    parser.add_argument("--model", default="glm-4.6v")
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max-table-chars", type=int, default=0, help="0 means no truncation.")
    parser.add_argument("--max-citekey-chars", type=int, default=0, help="0 means no truncation.")
    parser.add_argument("--max-ocr-texts", type=int, default=0, help="0 means no truncation.")
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true", help="Skip ids already present in the output JSONL.")
    return parser.parse_args()


def run_existing_prompts(args):
    prompt_rows = read_jsonl(args.input_prompts)
    selected = prompt_rows[args.start_index :]
    if args.max_samples > 0:
        selected = selected[: args.max_samples]

    if not args.run_model:
        summary = {
            "input_prompts": str(args.input_prompts),
            "prompt_kind": args.prompt_kind,
            "run_model": False,
            "total_prompt_rows": len(prompt_rows),
            "selected_samples": len(selected),
            "records_with_messages": sum(1 for row in selected if row.get("messages")),
            "records_with_system_prompt": sum(1 for row in selected if row.get("system_prompt")),
            "records_with_user_prompt": sum(1 for row in selected if row.get("user_prompt")),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    result_rows = []
    failures = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_jsonl = args.output_dir / f"ocr_glm46vflash_{args.prompt_kind}_model_results.jsonl"
    result_json = args.output_dir / f"ocr_glm46vflash_{args.prompt_kind}_model_results.json"
    full_result_jsonl = args.output_dir / f"ocr_glm46vflash_{args.prompt_kind}_model_results_full.jsonl"
    full_result_json = args.output_dir / f"ocr_glm46vflash_{args.prompt_kind}_model_results_full.json"
    progress_summary = args.output_dir / "summary.json"

    completed_ids = set()
    if args.resume and result_jsonl.exists():
        for old_row in read_jsonl(result_jsonl):
            old_id = old_row.get("id") or old_row.get("paper_id")
            if old_id:
                completed_ids.add(old_id)
    elif args.resume and full_result_jsonl.exists():
        for old_row in read_jsonl(full_result_jsonl):
            old_id = old_row.get("id") or old_row.get("paper_id")
            if old_id:
                completed_ids.add(old_id)

    def write_progress(processed_count):
        summary = {
            "input_prompts": str(args.input_prompts),
            "prompt_kind": args.prompt_kind,
            "total_prompt_rows": len(prompt_rows),
            "selected_samples": len(selected),
            "processed_this_run": processed_count,
            "completed_total_in_output": len(completed_ids),
            "api_base": args.api_base,
            "model": args.model,
            "match_threshold": args.match_threshold,
            "failures": failures,
            "result_jsonl": str(result_jsonl),
            "result_json": str(result_json),
            "full_result_jsonl": str(full_result_jsonl) if args.prompt_kind == "pairwise" else "",
            "full_result_json": str(full_result_json) if args.prompt_kind == "pairwise" else "",
        }
        write_json(progress_summary, summary)

    processed_this_run = 0

    for index, row in enumerate(selected, args.start_index):
        sample_id = row.get("id") or row.get("paper_id") or f"row_{index}"
        if args.resume and sample_id in completed_ids:
            logger.info("Skipping completed sample %s", sample_id)
            continue

        if args.prompt_kind == "pairwise":
            logger.info("Calling GLM pairwise for %s (%s/%s)", sample_id, index + 1, len(prompt_rows))
            citekey_evidences = split_citekey_evidences(
                row.get("prompt_inputs", {}).get("cited_facts_text", "")
            )
            sample_failures = []
            comparisons = []
            for evidence in citekey_evidences:
                system_prompt = build_system_prompt("pairwise")
                user_prompt = build_pairwise_comparison_prompt(row, evidence)
                try:
                    model_result = call_glm_text_model(
                        system_prompt,
                        user_prompt,
                        api_base=args.api_base,
                        api_key=args.api_key,
                        model=args.model,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    context = f"{sample_id}:{evidence.get('index')}"
                    parsed = robust_json_loads(extract_json(model_result), context=context)
                    match, score = normalize_match_score(
                        parsed,
                        context=context,
                        match_threshold=args.match_threshold,
                    )
                    comparison = {
                        "citekey_evidence": evidence,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "model_result": model_result,
                        "parsed_model_result": parsed,
                        "score": score,
                        "match": match,
                    }
                    comparisons.append(comparison)
                except Exception as exc:
                    logger.exception("Pairwise model call failed for %s evidence %s", sample_id, evidence.get("index"))
                    failure = {"id": sample_id, "citekey_evidence_index": evidence.get("index"), "error": str(exc)}
                    failures.append(failure)
                    sample_failures.append(failure)
                    comparisons.append(
                        {
                            "citekey_evidence": evidence,
                            "model_result": f"模型调用失败: {exc}",
                            "parsed_model_result": {},
                            "score": 0.0,
                            "match": False,
                        }
                    )
                if args.sleep > 0:
                    time.sleep(args.sleep)

            full_output = build_full_pairwise_output(
                row,
                comparisons,
                sample_failures,
                args.match_threshold,
            )
            output = build_clean_pairwise_output(
                row,
                comparisons,
                sample_failures,
                args.match_threshold,
            )
            append_jsonl(full_result_jsonl, full_output)
            append_jsonl(result_jsonl, output)
            result_rows.append(output)
            completed_ids.add(sample_id)
            processed_this_run += 1
            write_progress(processed_this_run)
            continue

        system_prompt = row.get("system_prompt", "")
        user_prompt = row.get("user_prompt") or row.get("prompt", "")
        if not system_prompt or not user_prompt:
            messages = row.get("messages") or []
            for message in messages:
                if message.get("role") == "system" and not system_prompt:
                    system_prompt = message.get("content", "")
                if message.get("role") == "user" and not user_prompt:
                    user_prompt = message.get("content", "")
        if not system_prompt or not user_prompt:
            failures.append({"id": sample_id, "error": "missing system_prompt or user_prompt"})
            continue

        logger.info("Calling GLM for %s (%s/%s)", sample_id, index + 1, len(prompt_rows))
        output = dict(row)
        try:
            model_result = call_glm_text_model(
                system_prompt,
                user_prompt,
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            parsed = robust_json_loads(extract_json(model_result), context=sample_id)
            output["model_result"] = model_result
            output["parsed_model_result"] = parsed

            bundle = {
                "tab_id": row.get("prompt_inputs", {}).get("tab_id", ""),
                "caption": row.get("prompt_inputs", {}).get("caption", ""),
                "table_rows_text": row.get("prompt_inputs", {}).get("table_rows_text", ""),
                "source_quality": row.get("prompt_inputs", {}).get("source_quality", {}),
            }
            output["originalkey_analysis"] = [convert_model_output_to_originalkey_analysis(bundle, parsed)]
        except Exception as exc:
            logger.exception("Model call failed for %s", sample_id)
            output["model_result"] = f"模型调用失败: {exc}"
            output["parsed_model_result"] = {}
            failures.append({"id": sample_id, "error": str(exc)})

        append_jsonl(result_jsonl, output)
        result_rows.append(output)
        completed_ids.add(sample_id)
        processed_this_run += 1
        write_progress(processed_this_run)
        if args.sleep > 0:
            time.sleep(args.sleep)

    if result_jsonl.exists():
        result_rows = read_jsonl(result_jsonl)
    write_json(result_json, result_rows)
    if args.prompt_kind == "pairwise" and full_result_jsonl.exists():
        write_json(full_result_json, read_jsonl(full_result_jsonl))

    summary = {
        "input_prompts": str(args.input_prompts),
        "prompt_kind": args.prompt_kind,
        "total_prompt_rows": len(prompt_rows),
        "selected_samples": len(selected),
        "model_result_records": len(result_rows),
        "processed_this_run": processed_this_run,
        "api_base": args.api_base,
        "model": args.model,
        "match_threshold": args.match_threshold,
        "failures": failures,
        "result_jsonl": str(result_jsonl),
        "result_json": str(result_json),
        "full_result_jsonl": str(full_result_jsonl) if args.prompt_kind == "pairwise" else "",
        "full_result_json": str(full_result_json) if args.prompt_kind == "pairwise" else "",
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    if args.input_prompts is not None:
        run_existing_prompts(args)
        return

    data = load_json_list(args.ocr_file)
    fallback_data = []
    if args.fallback_ocr_file and args.fallback_ocr_file.exists():
        fallback_data = load_json_list(args.fallback_ocr_file)
    fallback_by_id = {item.get("id") or item.get("paper_id"): item for item in fallback_data}

    selected = data[args.start_index :]
    if args.max_samples > 0:
        selected = selected[: args.max_samples]

    prompt_rows = []
    result_rows = []
    failures = []

    for index, item in enumerate(selected, args.start_index):
        bundle = build_bundle(
            item,
            fallback_by_id=fallback_by_id,
            max_table_chars=args.max_table_chars,
            max_ocr_texts=args.max_ocr_texts,
            max_citekey_chars=args.max_citekey_chars,
        )
        prompt = build_extraction_prompt(bundle)
        system_prompt = build_system_prompt(args.prompt_kind)
        prompt_record = build_output_record(item, bundle, system_prompt, prompt, args.prompt_kind)
        prompt_rows.append(prompt_record)

        if args.run_model:
            logger.info("Calling GLM for %s (%s/%s)", bundle["id"], index + 1, len(data))
            try:
                model_result = call_glm_text_model(
                    system_prompt,
                    prompt,
                    api_base=args.api_base,
                    api_key=args.api_key,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                parsed = robust_json_loads(extract_json(model_result), context=bundle["id"])
                result_rows.append(
                    build_output_record(
                        item,
                        bundle,
                        system_prompt,
                        prompt,
                        args.prompt_kind,
                        model_result,
                        parsed,
                    )
                )
            except Exception as exc:
                logger.exception("Model call failed for %s", bundle["id"])
                failures.append({"id": bundle["id"], "error": str(exc)})
                result_rows.append(
                    build_output_record(
                        item,
                        bundle,
                        system_prompt,
                        prompt,
                        args.prompt_kind,
                        model_result=f"模型调用失败: {exc}",
                        parsed={},
                    )
                )
            if args.sleep > 0:
                time.sleep(args.sleep)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_jsonl = args.output_dir / f"ocr_glm46vflash_{args.prompt_kind}_prompts.jsonl"
    prompt_json = args.output_dir / f"ocr_glm46vflash_{args.prompt_kind}_prompts.json"
    write_jsonl(prompt_jsonl, prompt_rows)
    write_json(prompt_json, prompt_rows)

    summary = {
        "ocr_file": str(args.ocr_file),
        "fallback_ocr_file": str(args.fallback_ocr_file),
        "prompt_kind": args.prompt_kind,
        "run_model": args.run_model,
        "total_input_samples": len(data),
        "selected_samples": len(selected),
        "prompt_records": len(prompt_rows),
        "records_with_pred_html": sum(
            1 for row in prompt_rows if row["prompt_inputs"]["source_quality"].get("has_pred_html")
        ),
        "records_with_candidates": sum(
            1
            for row in prompt_rows
            if row["prompt_inputs"]["source_quality"].get("num_candidate_rows", 0) > 0
        ),
        "records_using_fallback": sum(
            1
            for row in prompt_rows
            if row["prompt_inputs"]["source_quality"].get("used_fallback_originalkey_analysis")
        ),
        "failures": failures,
    }

    if args.run_model:
        result_jsonl = args.output_dir / f"ocr_glm46vflash_{args.prompt_kind}_model_results.jsonl"
        result_json = args.output_dir / f"ocr_glm46vflash_{args.prompt_kind}_model_results.json"
        write_jsonl(result_jsonl, result_rows)
        write_json(result_json, result_rows)
        summary["model_result_records"] = len(result_rows)
        summary["result_jsonl"] = str(result_jsonl)
        summary["result_json"] = str(result_json)

    summary["prompt_jsonl"] = str(prompt_jsonl)
    summary["prompt_json"] = str(prompt_json)
    write_json(args.output_dir / "summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
