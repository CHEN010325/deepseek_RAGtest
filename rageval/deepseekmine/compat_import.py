"""deepseekmine-compatible MinerU Markdown chunking and native KB upload.

This module keeps the evaluation bridge outside of deepseekmine. It mirrors the
core upload-route chunk document shape for QA sidecars, then uses deepseekmine's
native KB creation and file upload endpoints during real evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable

import requests


APP_ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_SAFE_TOKEN_LIMIT = 24000


def _estimate_dense_tokens(text: str) -> int:
    total = 0.0
    for char in text:
        if char.isspace():
            continue
        if "\u4e00" <= char <= "\u9fff":
            total += 1.5
        elif re.match(r"[A-Za-z0-9]", char):
            total += 0.45
        elif ord(char) < 128:
            total += 0.8
        else:
            total += 1
    return math.ceil(total)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_words = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))
    number_groups = len(re.findall(r"\d+(?:[.,:/_-]\d+)*", text))
    word_estimate = math.ceil(chinese_chars * 1.5 + latin_words * 1.3 + number_groups)
    return max(word_estimate, _estimate_dense_tokens(text))


def _split_by_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _split_by_lines(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n+", text) if part.strip()]


def _split_by_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.findall(r"[^。！？?!\n]+[。！？?!\n]*", text) if part.strip()]


def _split_by_words(text: str) -> list[str]:
    return [part.strip() for part in re.findall(r"\S+\s*", text) if part.strip()]


def _is_gfm_table_separator(line: str) -> bool:
    trimmed = line.strip()
    if "|" not in trimmed:
        return False
    cells = [cell.strip() for cell in trimmed.strip("|").split("|")]
    return len(cells) > 1 and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def _is_complete_html_table_block(text: str) -> bool:
    return bool(re.match(r"^<table\b[\s\S]*</table>\s*$", text.strip(), flags=re.I))


def _is_complete_gfm_table_block(text: str) -> bool:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    separator_index = next((idx for idx, line in enumerate(lines) if _is_gfm_table_separator(line)), -1)
    return separator_index >= 1 and all("|" in line for line in lines)


def _is_atomic_table_block(text: str) -> bool:
    normalized = text.strip()
    return _is_complete_html_table_block(normalized) or _is_complete_gfm_table_block(normalized)


def _hard_split_by_character_window(text: str, chunk_token_num: int) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    safe_char_window = max(200, int(chunk_token_num * 2))
    parts: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + safe_char_window)
        candidate = normalized[start:end].strip()
        while candidate and estimate_tokens(candidate) > chunk_token_num and end > start + 1:
            end -= max(1, (end - start) // 4)
            candidate = normalized[start:end].strip()
        if not candidate:
            end = min(len(normalized), start + 1)
            candidate = normalized[start:end].strip()
        if candidate:
            parts.append(candidate)
        start = end
    return parts


def _pack_segments_by_token_budget(
    segments: list[str],
    chunk_token_num: int,
    joiner: str,
    next_strategy_index: int,
    strategies: list[tuple[Callable[[str], list[str]], str]],
    preserve_atomic_tables: bool = True,
) -> list[str]:
    packed: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        normalized = current.strip()
        if normalized:
            packed.append(normalized)
        current = ""

    for segment in segments:
        normalized_segment = segment.strip()
        if not normalized_segment:
            continue
        if estimate_tokens(normalized_segment) > chunk_token_num:
            flush()
            packed.extend(
                _split_oversized_markdown_chunk(
                    normalized_segment,
                    chunk_token_num,
                    next_strategy_index,
                    strategies,
                    preserve_atomic_tables,
                )
            )
            continue
        candidate = f"{current}{joiner}{normalized_segment}" if current else normalized_segment
        if current and estimate_tokens(candidate) > chunk_token_num:
            flush()
            current = normalized_segment
        else:
            current = candidate
    flush()
    return packed


def _split_oversized_markdown_chunk(
    text: str,
    chunk_token_num: int,
    strategy_index: int = 0,
    strategies: list[tuple[Callable[[str], list[str]], str]] | None = None,
    preserve_atomic_tables: bool = True,
) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    if estimate_tokens(normalized) <= chunk_token_num:
        return [normalized]
    if preserve_atomic_tables and _is_atomic_table_block(normalized):
        return [normalized]
    strategies = strategies or [
        (_split_by_paragraphs, "\n\n"),
        (_split_by_lines, "\n"),
        (_split_by_sentences, "\n"),
        (_split_by_words, " "),
        (lambda value: list(value), ""),
    ]
    if strategy_index >= len(strategies):
        return _hard_split_by_character_window(normalized, chunk_token_num)
    splitter, joiner = strategies[strategy_index]
    segments = splitter(normalized)
    if len(segments) <= 1:
        return _split_oversized_markdown_chunk(
            normalized,
            chunk_token_num,
            strategy_index + 1,
            strategies,
            preserve_atomic_tables,
        )
    return _pack_segments_by_token_budget(
        segments,
        chunk_token_num,
        joiner,
        strategy_index + 1,
        strategies,
        preserve_atomic_tables,
    )


def _enforce_chunk_token_limit(chunks: list[str], chunk_token_num: int) -> list[str]:
    out: list[str] = []
    for chunk in chunks:
        out.extend(_split_oversized_markdown_chunk(chunk, chunk_token_num))
    return [chunk.strip() for chunk in out if chunk.strip()]


def _split_markdown_basic(text: str, chunk_token_num: int = 275, min_chunk_tokens: int = 50) -> list[str]:
    if _is_atomic_table_block(text):
        return [text.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for paragraph in [part for part in re.split(r"\n\s*\n", text) if part]:
        para_tokens = estimate_tokens(paragraph)
        if para_tokens > chunk_token_num:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0
            sentence_chunk: list[str] = []
            sentence_tokens = 0
            for sentence in re.split(r"(?<=[。！？?!])", paragraph):
                if not sentence:
                    continue
                sent_tokens = estimate_tokens(sentence)
                if sentence_tokens + sent_tokens > chunk_token_num and sentence_chunk:
                    chunks.append("".join(sentence_chunk))
                    sentence_chunk = [sentence]
                    sentence_tokens = sent_tokens
                else:
                    sentence_chunk.append(sentence)
                    sentence_tokens += sent_tokens
            if sentence_chunk:
                chunks.append("".join(sentence_chunk))
            continue
        if current and current_tokens + para_tokens > chunk_token_num:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_tokens = para_tokens
        else:
            current.append(paragraph)
            current_tokens += para_tokens
    if current and (current_tokens >= min_chunk_tokens or not chunks):
        chunks.append("\n\n".join(current))
    filtered = _enforce_chunk_token_limit(chunks, chunk_token_num)
    if not filtered and text.strip():
        return _split_oversized_markdown_chunk(text.strip(), chunk_token_num)
    return filtered


def _split_markdown_by_title(text: str, chunk_token_num: int = 275, min_chunk_tokens: int = 50) -> list[str]:
    matches = list(re.finditer(r"^(#{1,6})\s+(.+)$", text, flags=re.M))
    if not matches:
        return _split_markdown_basic(text, chunk_token_num, min_chunk_tokens)
    chunks: list[str] = []
    first_title_start = matches[0].start()
    if first_title_start > 0:
        prefix = text[:first_title_start].strip()
        if prefix:
            prefix_tokens = estimate_tokens(prefix)
            if prefix_tokens > chunk_token_num:
                chunks.extend(_split_markdown_basic(prefix, chunk_token_num, min_chunk_tokens))
            elif prefix_tokens >= min_chunk_tokens or not chunks:
                chunks.append(prefix)
    for index, match in enumerate(matches):
        section_start = match.start()
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[section_start:section_end].strip()
        section_tokens = estimate_tokens(section)
        if section_tokens > chunk_token_num:
            para_chunk: list[str] = []
            para_tokens = 0
            for paragraph in re.split(r"\n\s*\n", section):
                paragraph_tokens = estimate_tokens(paragraph)
                if para_chunk and para_tokens + paragraph_tokens > chunk_token_num:
                    chunks.append("\n\n".join(para_chunk))
                    para_chunk = [paragraph]
                    para_tokens = paragraph_tokens
                else:
                    para_chunk.append(paragraph)
                    para_tokens += paragraph_tokens
            if para_chunk:
                chunks.append("\n\n".join(para_chunk))
        elif section_tokens >= min_chunk_tokens or not chunks:
            chunks.append(section)
    filtered = _enforce_chunk_token_limit(chunks, chunk_token_num)
    if not filtered and text.strip():
        return _split_oversized_markdown_chunk(text.strip(), chunk_token_num)
    return filtered


def _normalize_for_search(value: str) -> tuple[str, list[int]]:
    text = ""
    index_map: list[int] = []
    prev_whitespace = False
    for index, char in enumerate(value):
        if char.isspace():
            if not prev_whitespace:
                text += " "
                index_map.append(index)
                prev_whitespace = True
            continue
        text += char
        index_map.append(index)
        prev_whitespace = False
    return text, index_map


def _locate_markdown_chunk_ranges(text: str, chunks: list[str]) -> list[dict[str, Any]]:
    exact: list[dict[str, Any]] = []
    cursor = 0
    exact_success = True
    for content in chunks:
        exact_index = text.find(content, cursor)
        if exact_index < 0:
            exact_success = False
            break
        exact.append({"content": content, "start": exact_index, "end": exact_index + len(content)})
        cursor = exact_index + len(content)
    if exact_success:
        return exact

    normalized_source, index_map = _normalize_for_search(text)
    results: list[dict[str, Any]] = []
    normalized_cursor = 0
    for content in chunks:
        normalized_chunk, _ = _normalize_for_search(content)
        normalized_chunk = normalized_chunk.strip()
        if not normalized_chunk:
            results.append({"content": content, "start": -1, "end": -1})
            continue
        normalized_index = normalized_source.find(normalized_chunk, normalized_cursor)
        if normalized_index < 0:
            results.append({"content": content, "start": -1, "end": -1})
            continue
        start = index_map[normalized_index] if normalized_index < len(index_map) else -1
        end_map_index = normalized_index + len(normalized_chunk) - 1
        end = (index_map[end_map_index] + 1) if 0 <= end_map_index < len(index_map) else start + len(content)
        results.append({"content": content, "start": start, "end": end})
        normalized_cursor = normalized_index + len(normalized_chunk)
    return results


def split_chunks_with_metadata(text: str, file_type: str = "md") -> list[dict[str, Any]]:
    if file_type.lower() in {"md", "markdown"}:
        chunk_token_num = 1000
        text_tokens = estimate_tokens(text)
        default_min_chunk_tokens = max(50, 275 / 5)
        min_chunk_tokens = min(10, max(1, text_tokens / 2)) if text_tokens < default_min_chunk_tokens else default_min_chunk_tokens
        chunks = _split_markdown_by_title(text, int(chunk_token_num), int(min_chunk_tokens))
        return _locate_markdown_chunk_ranges(text, chunks)
    chunks = _split_markdown_basic(text, 275, 55)
    return _locate_markdown_chunk_ranges(text, chunks)


def _trim_upload_chunk(raw_text: str, start: int, end: int) -> dict[str, Any] | None:
    left = max(0, min(len(raw_text), int(start)))
    right = max(left, min(len(raw_text), int(end)))
    while left < right and raw_text[left].isspace():
        left += 1
    while right > left and raw_text[right - 1].isspace():
        right -= 1
    if right <= left:
        return None
    return {"content": raw_text[left:right], "start": left, "end": right}


def _is_heading_only_chunk(content: str) -> bool:
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    return len(lines) == 1 and bool(re.match(r"^#{1,6}\s+\S", lines[0]))


def _merge_heading_only_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    index = 0
    while index < len(chunks):
        current = chunks[index]
        next_chunk = chunks[index + 1] if index + 1 < len(chunks) else None
        if next_chunk and _is_heading_only_chunk(str(current.get("content") or "")):
            merged.append(
                {
                    "content": f"{str(current['content']).strip()}\n\n{str(next_chunk['content']).strip()}",
                    "start": current["start"],
                    "end": next_chunk["end"],
                }
            )
            index += 2
            continue
        merged.append(current)
        index += 1
    return merged


def _split_text_range_by_original_markdown(raw_text: str, start: int, end: int, file_type: str) -> list[dict[str, Any]]:
    trimmed = _trim_upload_chunk(raw_text, start, end)
    if not trimmed:
        return []
    chunks = []
    for chunk in split_chunks_with_metadata(trimmed["content"], file_type):
        chunks.append(
            {
                "content": chunk["content"],
                "start": trimmed["start"] + max(0, int(chunk["start"])),
                "end": trimmed["start"] + max(0, int(chunk["end"])),
            }
        )
    return _merge_heading_only_chunks(chunks) if file_type.lower() == "md" else chunks


def _expand_image_span(raw_text: str, span: dict[str, Any]) -> tuple[int, int]:
    start = max(0, min(len(raw_text), int(span.get("start", 0))))
    end = max(start, min(len(raw_text), int(span.get("end", start))))
    look_behind_start = max(0, start - 2000)
    look_behind = raw_text[look_behind_start:start]
    match = re.search(
        r"(?:^|\r?\n)[ \t]*(?:!\[[^\]\r\n]*\]\([^)]+\)|<img\b[\s\S]*?>)[ \t]*(?:\r?\n[ \t]*)*$",
        look_behind,
        flags=re.I,
    )
    if match:
        start = look_behind_start + match.start()
    return start, end


def _trim_table_span_before_trailing_image(raw_text: str, start: int, end: int) -> tuple[int, int]:
    text_slice = raw_text[start:end]
    match = re.search(
        r"(?:\r?\n[ \t]*)+(?:!\[[^\]\r\n]*\]\([^)]+\)|<img\b[\s\S]*?>)[ \t]*(?:\r?\n[ \t]*)*$",
        text_slice,
        flags=re.I,
    )
    if not match:
        return start, end
    if not re.search(r"<table\b", text_slice[: match.start()], flags=re.I):
        return start, end
    return start, start + match.start()


def _normalize_atomic_semantic_span(raw_text: str, span: dict[str, Any]) -> dict[str, Any] | None:
    span_type = str(span.get("type") or "").lower()
    if span_type not in {"table", "image"}:
        return None
    initial_start = max(0, min(len(raw_text), int(span.get("start", 0))))
    initial_end = max(initial_start, min(len(raw_text), int(span.get("end", initial_start))))
    expanded = (
        _trim_table_span_before_trailing_image(raw_text, initial_start, initial_end)
        if span_type == "table"
        else _expand_image_span(raw_text, span)
    )
    chunk = _trim_upload_chunk(raw_text, expanded[0], expanded[1])
    if chunk:
        chunk["atomicType"] = span_type
    return chunk


def _clamp_span_to_markdown_section(raw_text: str, span: dict[str, Any]) -> dict[str, Any] | None:
    section_start = 0
    section_end = len(raw_text)
    for match in re.finditer(r"^#{1,6}\s+\S.*$", raw_text, flags=re.M):
        heading_start = match.start()
        if heading_start <= span["start"]:
            section_start = heading_start
            continue
        section_end = heading_start
        break
    trimmed = _trim_upload_chunk(raw_text, max(section_start, span["start"]), min(section_end, span["end"]))
    if trimmed:
        trimmed["atomicType"] = span.get("atomicType")
    return trimmed


def _get_ordered_atomic_semantic_spans(raw_text: str, semantic_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for span in semantic_spans:
        try:
            if int(span.get("end", 0)) <= int(span.get("start", 0)):
                continue
            normalized = _normalize_atomic_semantic_span(raw_text, span)
            if not normalized:
                continue
            clamped = _clamp_span_to_markdown_section(raw_text, normalized)
            if clamped:
                spans.append(clamped)
        except Exception:
            continue
    return sorted(spans, key=lambda item: int(item["start"]))


def _merge_markdown_chunks_around_atomic_spans(
    raw_text: str,
    markdown_chunks: list[dict[str, Any]],
    atomic_spans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chunks = sorted(markdown_chunks, key=lambda item: int(item["start"]))
    for span in atomic_spans:
        first = next((idx for idx, chunk in enumerate(chunks) if chunk["end"] > span["start"] and chunk["start"] < span["end"]), -1)
        if first < 0:
            insert_at = next((idx for idx, chunk in enumerate(chunks) if chunk["start"] > span["start"]), len(chunks))
            chunks.insert(insert_at, span)
            continue
        last = first
        while last + 1 < len(chunks) and chunks[last + 1]["start"] < span["end"]:
            last += 1
        merged = _trim_upload_chunk(raw_text, min(chunks[first]["start"], span["start"]), max(chunks[last]["end"], span["end"]))
        if merged:
            chunks[first : last + 1] = [merged]
    return chunks


def _split_by_semantic_spans(raw_text: str, semantic_spans: list[dict[str, Any]], file_type: str) -> list[dict[str, Any]]:
    markdown_chunks = _split_text_range_by_original_markdown(raw_text, 0, len(raw_text), file_type)
    atomic_spans = _get_ordered_atomic_semantic_spans(raw_text, semantic_spans)
    if not atomic_spans:
        return markdown_chunks
    merged = _merge_markdown_chunks_around_atomic_spans(raw_text, markdown_chunks, atomic_spans)
    return merged or markdown_chunks


def _load_mineru_metadata(markdown_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = [
        markdown_path.with_suffix(markdown_path.suffix + ".meta.json"),
        markdown_path.with_name(markdown_path.name + ".meta.json"),
        markdown_path.with_suffix(".meta.json"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        page_spans = payload.get("page_spans") or payload.get("pageSpans") or []
        semantic_spans = payload.get("semantic_spans") or payload.get("semanticSpans") or []
        return _normalize_spans(page_spans, True), _normalize_spans(semantic_spans, False)
    return [], []


def _normalize_spans(spans: Any, page_required: bool) -> list[dict[str, Any]]:
    if not isinstance(spans, list):
        return []
    out: list[dict[str, Any]] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        try:
            start = int(span.get("start"))
            end = int(span.get("end"))
        except Exception:
            continue
        if end <= start:
            continue
        normalized = {"start": start, "end": end}
        if span.get("page") is not None:
            try:
                normalized["page"] = int(span.get("page"))
            except Exception:
                pass
        elif page_required:
            continue
        span_type = str(span.get("type") or "").strip()
        if span_type:
            normalized["type"] = span_type
        title = str(span.get("title") or "").strip()
        if title:
            normalized["title"] = title
        out.append(normalized)
    return out


def _get_chunk_page_value(page_spans: list[dict[str, Any]], start: int, end: int) -> str:
    if not page_spans or start < 0 or end <= start:
        return ""
    pages: set[int] = set()
    for span in page_spans:
        if int(span["end"]) > start and int(span["start"]) < end and span.get("page") is not None:
            pages.add(int(span["page"]))
    ordered = sorted(pages)
    if not ordered:
        return ""
    return str(ordered[0]) if ordered[0] == ordered[-1] else f"{ordered[0]}-{ordered[-1]}"


def _split_oversized_table_for_embedding(text: str, chunk_token_num: int) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    if not _is_atomic_table_block(normalized) or estimate_tokens(normalized) <= chunk_token_num:
        return [normalized]
    return _split_oversized_markdown_chunk(normalized, chunk_token_num, preserve_atomic_tables=False)


def build_mineru_markdown_chunk_docs(
    markdown_text: str,
    doc_index_base: str,
    knowledge_label: str,
    page_spans: list[dict[str, Any]] | None = None,
    semantic_spans: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    page_spans = page_spans or []
    semantic_spans = semantic_spans or []
    chunk_metas = (
        _split_by_semantic_spans(markdown_text, semantic_spans, "md")
        if semantic_spans
        else _split_text_range_by_original_markdown(markdown_text, 0, len(markdown_text), "md")
    )
    docs: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunk_metas, start=1):
        original = str(chunk.get("content") or "")
        if not original.strip():
            continue
        stable_id = hashlib.sha1(f"{knowledge_label}::{doc_index_base}::{idx}".encode("utf-8")).hexdigest()
        page = _get_chunk_page_value(page_spans, int(chunk["start"]), int(chunk["end"]))
        embedding_parts = _split_oversized_table_for_embedding(original, EMBEDDING_SAFE_TOKEN_LIMIT)
        is_oversized_table = len(embedding_parts) > 1
        source_doc: dict[str, Any] = {
            "id": stable_id,
            "doc_index2": idx,
            "title": doc_index_base,
            "content": original,
        }
        if is_oversized_table:
            source_doc.update(
                {
                    "embeddingContent": embedding_parts[0] or original,
                    "fragmentRole": "display",
                    "sourceStart": int(chunk["start"]),
                    "sourceEnd": int(chunk["end"]),
                }
            )
        if page:
            source_doc["page"] = page
        docs.append(source_doc)
        if not is_oversized_table:
            continue
        for part_index, part in enumerate(embedding_parts[1:], start=2):
            part_doc: dict[str, Any] = {
                "id": hashlib.sha1(
                    f"{knowledge_label}::{doc_index_base}::{idx}::embedding::{part_index}".encode("utf-8")
                ).hexdigest(),
                "doc_index2": idx + part_index / 10000,
                "title": doc_index_base,
                "content": part,
                "fragmentRole": "embedding_part",
                "sourceDocIndex2": idx,
                "sourceStart": int(chunk["start"]),
                "sourceEnd": int(chunk["end"]),
                "tablePart": part_index,
                "tablePartTotal": len(embedding_parts),
            }
            if page:
                part_doc["page"] = page
            docs.append(part_doc)
    return docs


def build_chunk_docs_from_markdown_path(markdown_path: Path, knowledge_label: str) -> list[dict[str, Any]]:
    markdown_text = markdown_path.read_text(encoding="utf-8", errors="ignore")
    page_spans, semantic_spans = _load_mineru_metadata(markdown_path)
    return build_mineru_markdown_chunk_docs(markdown_text, markdown_path.name, knowledge_label, page_spans, semantic_spans)


def write_chunk_sidecar(markdown_path: Path, dataset_name: str) -> Path | None:
    if not markdown_path.exists() or not markdown_path.with_suffix(markdown_path.suffix + ".mineru.json").exists():
        return None
    docs = build_chunk_docs_from_markdown_path(markdown_path, f"dryrun_{dataset_name}")
    chunks = [
        {
            "doc_index2": item.get("doc_index2"),
            "title": item.get("title"),
            "content": item.get("content"),
            "page": item.get("page", ""),
            "fragmentRole": item.get("fragmentRole", ""),
            "sourceStart": item.get("sourceStart"),
            "sourceEnd": item.get("sourceEnd"),
            "sourceDocIndex2": item.get("sourceDocIndex2"),
            "tablePart": item.get("tablePart"),
            "tablePartTotal": item.get("tablePartTotal"),
        }
        for item in docs
        if str(item.get("content") or "").strip()
    ]
    if not chunks:
        return None
    sidecar = markdown_path.with_suffix(markdown_path.suffix + ".deepseekmine_chunks.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema": "deepseekmine-mineru-chunks-v1",
                "source": "rageval-local-deepseekmine-compatible",
                "chunks": chunks,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return sidecar


def create_native_knowledge_base(api_base: str, title: str, timeout: int = 60) -> dict[str, Any]:
    resp = requests.post(
        api_base.rstrip("/") + "/api/kb",
        json={
            "title": title,
            "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "creator": "RAGEval Forge",
            "icon": "RF",
            "bgColor": "#0f766e",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    created = payload.get("created") if isinstance(payload, dict) else None
    if not isinstance(created, dict) or created.get("id") is None:
        raise RuntimeError(f"deepseekmine create KB failed: {payload}")
    return created


def upload_markdown_assets_via_native_upload(
    api_base: str,
    knowledge_title: str,
    markdown_paths: list[Path],
    timeout: int = 900,
    progress: Callable[[str], None] | None = None,
) -> tuple[str, list[str], dict[str, Any]]:
    """Create a native KB, then upload Markdown files through /api/files/upload.

    Passing localPath lets deepseekmine's existing upload route discover the
    adjacent MinerU .meta.json sidecar and run its own production chunker.
    """

    log = progress or (lambda _msg: None)
    unique_title = f"{knowledge_title}_{int(time.time())}"
    created = create_native_knowledge_base(api_base, unique_title, min(timeout, 60))
    kb_id = str(created["id"])
    time.sleep(1.0)

    fields: list[tuple[str, str]] = [
        ("knowledgeLabel", kb_id),
        ("waitForProcessing", "true"),
        ("fastParsing", "true"),
    ]
    file_handles = []
    files = []
    try:
        for markdown_path in markdown_paths:
            fields.append(("relativePath", markdown_path.name))
            fields.append(("localPath", str(markdown_path)))
            handle = markdown_path.open("rb")
            file_handles.append(handle)
            files.append(("file", (markdown_path.name, handle, "text/markdown")))
        log(f"上传 MinerU Markdown 到 deepseekmine 原生上传接口：knowledgeLabel={kb_id}")
        resp = requests.post(
            api_base.rstrip("/") + "/api/files/upload",
            data=fields,
            files=files,
            timeout=timeout,
        )
        resp.raise_for_status()
    finally:
        for handle in file_handles:
            try:
                handle.close()
            except Exception:
                pass

    payload = resp.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"deepseekmine native markdown upload failed: {payload}")
    doc_ids = [str(x).strip() for x in payload.get("files", []) if str(x).strip()]
    if not doc_ids:
        doc_ids = [path.name for path in markdown_paths]
    upload_payload = {
        "code": 0,
        "method": "native_files_upload_with_mineru_sidecar",
        "requestedKnowledgeLabel": knowledge_title,
        "importedKnowledgeLabel": kb_id,
        "importedKnowledgeTitle": unique_title,
        "createdKnowledgeBase": created,
        "files": doc_ids,
        "uploadResponse": payload,
    }
    return kb_id, doc_ids, upload_payload
