"""
Build auditable RAG evaluation datasets from local documents.

The output is JSONL for auditable RAG evaluation: each line has
id/query/answer/evidence. Legacy positive/negative fields are kept for
existing E:\\RAG evaluators, with positive mapped to the supporting
evidence quote and negative empty by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import random
import re
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable
from xml.etree import ElementTree as ET

import requests

from rageval.llm import (
    LLMQuotaExceededError as ProviderQuotaExceededError,
    chat_completion,
    default_provider_config,
    load_llm_config,
    message_text,
)


QUOTA_ERROR_PATTERNS = [
    "insufficient_quota",
    "quota exceeded",
    "quota_exceeded",
    "quota exhausted",
    "insufficient balance",
    "out of credits",
    "billing",
    "payment required",
    "account balance",
    "resource exhausted",
    "request limit",
    "余额不足",
    "额度不足",
    "配额不足",
    "账户余额",
    "资源耗尽",
    "欠费",
]


class LLMQuotaExceededError(RuntimeError):
    pass


def is_quota_error(status_code: int, body: str) -> bool:
    text = str(body or "").lower()
    if status_code == 402:
        return True
    return any(pattern in text for pattern in QUOTA_ERROR_PATTERNS)


SUPPORTED_UPLOAD_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".txt",
    ".md",
    ".markdown",
    ".canvas",
    ".log",
    ".eml",
    ".epub",
    ".xls",
    ".xlsx",
    ".csv",
    ".json",
    ".jsonl",
    ".py",
    ".js",
    ".ts",
    ".html",
    ".htm",
    ".css",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
    ".gif",
    ".heic",
    ".wps",
    ".wpt",
    ".et",
    ".ett",
}
ARCHIVE_EXTENSIONS = {".zip"}
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".canvas",
    ".log",
    ".py",
    ".js",
    ".ts",
    ".css",
    ".json",
    ".jsonl",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif", ".gif", ".heic"}


@dataclass
class ParsedDocument:
    source_file: str
    text: str
    file_type: str
    parser: str
    page_spans: list[dict[str, int]]
    semantic_spans: list[dict[str, Any]] | None = None


@dataclass
class TextBlock:
    block_id: str
    source_file: str
    file_type: str
    content: str
    start: int
    end: int
    page: str


@dataclass
class Evidence:
    source_file: str
    file_type: str
    chunk_id: str
    page: str
    quote: str
    char_start: int
    char_end: int


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag.lower() in {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag.lower() in {"p", "div", "li", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        return normalize_text(html.unescape(" ".join(self.parts)))


def normalize_text(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    value = re.sub(r"[ \t\f\v]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def normalize_for_match(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def safe_decode(data: bytes) -> str:
    if not data:
        return ""
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "big5", "latin1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def strip_html(raw: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(raw)
    return parser.text()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def xml_text(xml_bytes: bytes) -> str:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return ""
    parts: list[str] = []
    for elem in root.iter():
        tag = local_name(elem.tag)
        if tag in {"t", "instrText"} and elem.text:
            parts.append(elem.text)
        elif tag in {"tab"}:
            parts.append("\t")
        elif tag in {"br", "cr", "p", "tr"}:
            parts.append("\n")
        elif tag in {"tc"}:
            parts.append("\t")
    return normalize_text("".join(parts))


def parse_docx(data: bytes, source: str) -> ParsedDocument:
    with zipfile.ZipFile(io_bytes(data)) as zf:
        document_names = [n for n in zf.namelist() if n.lower() == "word/document.xml"]
        if not document_names:
            raise ValueError("docx missing word/document.xml")
        text = xml_text(zf.read(document_names[0]))
    return parsed(source, text, ".docx", "docx-xml")


def parse_pptx(data: bytes, source: str) -> ParsedDocument:
    slide_texts: list[str] = []
    spans: list[dict[str, int]] = []
    cursor = 0
    with zipfile.ZipFile(io_bytes(data)) as zf:
        slide_names = sorted(
            [n for n in zf.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n, re.I)],
            key=lambda n: int(re.search(r"slide(\d+)\.xml$", n, re.I).group(1)),
        )
        for slide_idx, name in enumerate(slide_names, 1):
            part = xml_text(zf.read(name))
            if not part:
                continue
            if slide_texts:
                slide_texts.append("\n\n")
                cursor += 2
            start = cursor
            slide_texts.append(part)
            cursor += len(part)
            spans.append({"page": slide_idx, "start": start, "end": cursor})
    return ParsedDocument(source, normalize_text("".join(slide_texts)), ".pptx", "pptx-xml", spans)


def parse_xlsx(data: bytes, source: str, ext: str) -> ParsedDocument:
    with zipfile.ZipFile(io_bytes(data)) as zf:
        shared_strings = load_shared_strings(zf)
        sheet_names = load_workbook_sheet_names(zf)
        sheet_files = sorted(
            [n for n in zf.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n, re.I)],
            key=lambda n: int(re.search(r"sheet(\d+)\.xml$", n, re.I).group(1)),
        )
        blocks: list[str] = []
        for idx, name in enumerate(sheet_files):
            sheet_title = sheet_names[idx] if idx < len(sheet_names) else f"Sheet{idx + 1}"
            rows = parse_sheet_rows(zf.read(name), shared_strings)
            for row_number, row in rows:
                values = [cell for cell in row if cell]
                if values:
                    blocks.append(f"{Path(source).name} - {sheet_title} - row {row_number}\n" + "\n".join(values))
    return parsed(source, "\n\n".join(blocks), ext, "xlsx-xml")


def load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except Exception:
        return []
    strings: list[str] = []
    for si in root.iter():
        if local_name(si.tag) != "si":
            continue
        parts = [t.text or "" for t in si.iter() if local_name(t.tag) == "t"]
        strings.append("".join(parts))
    return strings


def load_workbook_sheet_names(zf: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(zf.read("xl/workbook.xml"))
    except Exception:
        return []
    names: list[str] = []
    for sheet in root.iter():
        if local_name(sheet.tag) == "sheet":
            name = sheet.attrib.get("name")
            if name:
                names.append(name)
    return names


def parse_sheet_rows(xml_bytes_: bytes, shared_strings: list[str]) -> list[tuple[int, list[str]]]:
    try:
        root = ET.fromstring(xml_bytes_)
    except ET.ParseError:
        return []
    rows: list[tuple[int, list[str]]] = []
    for row in root.iter():
        if local_name(row.tag) != "row":
            continue
        row_number = int(row.attrib.get("r", len(rows) + 1))
        cells: list[str] = []
        for cell in row:
            if local_name(cell.tag) != "c":
                continue
            cell_type = cell.attrib.get("t")
            raw_value = ""
            for child in cell:
                if local_name(child.tag) == "v":
                    raw_value = child.text or ""
                    break
                if local_name(child.tag) == "is":
                    raw_value = "".join(t.text or "" for t in child.iter() if local_name(t.tag) == "t")
                    break
            if cell_type == "s" and raw_value.isdigit():
                idx = int(raw_value)
                raw_value = shared_strings[idx] if idx < len(shared_strings) else raw_value
            cells.append(str(raw_value).strip())
        if cells:
            rows.append((row_number, cells))
    return rows


def parse_epub(data: bytes, source: str) -> ParsedDocument:
    parts: list[str] = []
    with zipfile.ZipFile(io_bytes(data)) as zf:
        html_names = [
            n for n in zf.namelist()
            if n.lower().endswith((".xhtml", ".html", ".htm")) and not n.endswith("/")
        ]
        for name in sorted(html_names):
            parts.append(strip_html(safe_decode(zf.read(name))))
    return parsed(source, "\n\n".join(parts), ".epub", "epub-zip-html")


def parse_csv(data: bytes, source: str, ext: str = ".csv") -> ParsedDocument:
    text = safe_decode(data)
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        return parsed(source, "", ext, "csv")
    header = rows[0]
    blocks: list[str] = []
    for index, row in enumerate(rows[1:] if len(rows) > 1 else rows, 1):
        if header and len(header) == len(row):
            fields = [f"{h}: {v}" for h, v in zip(header, row) if str(v).strip()]
        else:
            fields = [str(v) for v in row if str(v).strip()]
        if fields:
            blocks.append(f"{Path(source).name} - row {index}\n" + "\n".join(fields))
    return parsed(source, "\n\n".join(blocks), ext, "csv")


def parse_eml(data: bytes, source: str) -> ParsedDocument:
    message = BytesParser(policy=policy.default).parsebytes(data)
    parts: list[str] = []
    subject = message.get("subject")
    if subject:
        parts.append(f"Subject: {subject}")
    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                parts.append(str(part.get_content()))
            elif content_type == "text/html":
                parts.append(strip_html(str(part.get_content())))
    else:
        payload = message.get_content()
        parts.append(strip_html(str(payload)) if message.get_content_type() == "text/html" else str(payload))
    return parsed(source, "\n\n".join(parts), ".eml", "email")


def parse_pdf(data: bytes, source: str) -> ParsedDocument:
    errors: list[str] = []
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(io_bytes(data))
        parts: list[str] = []
        spans: list[dict[str, int]] = []
        cursor = 0
        for page_index, page in enumerate(reader.pages, 1):
            page_text = page.extract_text() or ""
            page_text = normalize_text(page_text)
            if not page_text:
                continue
            if parts:
                parts.append("\n\n")
                cursor += 2
            start = cursor
            parts.append(page_text)
            cursor += len(page_text)
            spans.append({"page": page_index, "start": start, "end": cursor})
        text = normalize_text("".join(parts))
        if text:
            return ParsedDocument(source, text, ".pdf", "pypdf", spans)
    except Exception as exc:
        errors.append(f"pypdf: {exc}")

    try:
        import PyPDF2  # type: ignore

        reader = PyPDF2.PdfReader(io_bytes(data))
        text = "\n\n".join(normalize_text(page.extract_text() or "") for page in reader.pages)
        if text.strip():
            return parsed(source, text, ".pdf", "PyPDF2")
    except Exception as exc:
        errors.append(f"PyPDF2: {exc}")

    raise ValueError("PDF text extraction failed; install pypdf/PyPDF2 or OCR scanned PDFs first. " + "; ".join(errors))


def io_bytes(data: bytes):
    import io

    return io.BytesIO(data)


def parsed(source: str, text: str, ext: str, parser_name: str) -> ParsedDocument:
    text = normalize_text(text)
    spans = [{"page": 1, "start": 0, "end": len(text)}] if text else []
    sidecar = load_mineru_sidecar_metadata(source)
    page_spans = sidecar.get("page_spans") or sidecar.get("pageSpans") or spans
    semantic_spans = sidecar.get("semantic_spans") or sidecar.get("semanticSpans") or None
    parser = f"{parser_name}+mineru-semantic" if semantic_spans else parser_name
    return ParsedDocument(source, text, ext, parser, page_spans, semantic_spans)


def load_mineru_sidecar_metadata(source: str) -> dict[str, Any]:
    path = Path(source)
    if not path.exists() or path.suffix.lower() not in {".md", ".markdown"}:
        return {}
    candidates = [
        path.with_suffix(path.suffix + ".meta.json"),
        path.with_name(path.name + ".meta.json"),
        path.with_suffix(".meta.json"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        page_spans = normalize_spans(data.get("page_spans") or data.get("pageSpans"), page_required=True)
        semantic_spans = normalize_spans(data.get("semantic_spans") or data.get("semanticSpans"), page_required=False)
        if page_spans or semantic_spans:
            return {"page_spans": page_spans, "semantic_spans": semantic_spans}
    return {}


def normalize_spans(value: Any, page_required: bool) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    spans: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            start = int(item.get("start"))
            end = int(item.get("end"))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        span: dict[str, Any] = {"start": start, "end": end}
        page_value = item.get("page")
        try:
            page = int(page_value)
        except (TypeError, ValueError):
            page = None
        if page is not None:
            span["page"] = page
        elif page_required:
            continue
        if item.get("type"):
            span["type"] = str(item.get("type"))
        if item.get("title"):
            span["title"] = str(item.get("title"))
        spans.append(span)
    return spans


def parse_bytes(data: bytes, source: str) -> ParsedDocument:
    ext = Path(source).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(data, source)
    if ext in {".docx", ".wps", ".wpt"} and data[:2] == b"PK":
        return parse_docx(data, source)
    if ext == ".pptx":
        return parse_pptx(data, source)
    if ext in {".xlsx", ".et", ".ett"} and data[:2] == b"PK":
        return parse_xlsx(data, source, ext)
    if ext == ".csv":
        return parse_csv(data, source)
    if ext == ".eml":
        return parse_eml(data, source)
    if ext == ".epub":
        return parse_epub(data, source)
    if ext in {".html", ".htm"}:
        return parsed(source, strip_html(safe_decode(data)), ext, "html")
    if ext in TEXT_EXTENSIONS:
        return parsed(source, safe_decode(data), ext, "text")
    if ext in {".doc", ".ppt", ".xls"}:
        return parsed(source, safe_decode(data), ext, "binary-text-fallback")
    if ext in IMAGE_EXTENSIONS:
        raise ValueError("image OCR is not available in this standalone builder; OCR to PDF/MD/TXT first")
    raise ValueError(f"unsupported extension: {ext}")


def iter_input_payloads(paths: list[Path], max_zip_depth: int = 3) -> Iterable[tuple[str, bytes]]:
    for path in paths:
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and is_supported(child.name):
                    yield from iter_input_payloads([child], max_zip_depth=max_zip_depth)
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        data = path.read_bytes()
        if path.suffix.lower() in ARCHIVE_EXTENSIONS:
            yield from expand_zip_payload(data, path.name, max_zip_depth)
        elif is_supported(path.name, allow_archive=False):
            yield (str(path), data)


def expand_zip_payload(data: bytes, archive_name: str, depth: int) -> Iterable[tuple[str, bytes]]:
    if depth <= 0:
        return
    with zipfile.ZipFile(io_bytes(data)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            source_name = f"{archive_name}/{name}"
            ext = Path(name).suffix.lower()
            payload = zf.read(info)
            if ext in ARCHIVE_EXTENSIONS:
                yield from expand_zip_payload(payload, source_name, depth - 1)
            elif is_supported(name, allow_archive=False):
                yield (source_name, payload)


def is_supported(name: str, allow_archive: bool = True) -> bool:
    ext = Path(name).suffix.lower()
    return ext in SUPPORTED_UPLOAD_EXTENSIONS or (allow_archive and ext in ARCHIVE_EXTENSIONS)


def page_for_span(page_spans: list[dict[str, int]], start: int, end: int) -> str:
    pages = [s["page"] for s in page_spans if s.get("end", 0) > start and s.get("start", 0) < end]
    if not pages:
        return ""
    return str(min(pages)) if min(pages) == max(pages) else f"{min(pages)}-{max(pages)}"


def split_blocks(doc: ParsedDocument, max_chars: int, min_chars: int, overlap: int) -> list[TextBlock]:
    deepseekmine_blocks = load_deepseekmine_chunk_blocks(doc)
    if deepseekmine_blocks:
        return deepseekmine_blocks
    if is_markdown_document(doc):
        if doc.semantic_spans:
            semantic_blocks = split_markdown_with_semantic_spans(doc, max_chars, min_chars, overlap)
            if semantic_blocks:
                return semantic_blocks
        markdown_blocks = split_markdown_heading_blocks(doc, max_chars, min_chars, overlap)
        if markdown_blocks:
            return markdown_blocks
    return split_paragraph_blocks(doc, max_chars, min_chars, overlap)


def is_markdown_document(doc: ParsedDocument) -> bool:
    ext = Path(doc.source_file).suffix.lower()
    return doc.file_type in {".md", ".markdown"} or ext in {".md", ".markdown"}


def load_deepseekmine_chunk_blocks(doc: ParsedDocument) -> list[TextBlock]:
    path = Path(doc.source_file)
    if not path.exists():
        return []
    sidecar = path.with_suffix(path.suffix + ".deepseekmine_chunks.json")
    if not sidecar.exists():
        return []
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw_chunks = payload.get("chunks") if isinstance(payload, dict) else None
    if not isinstance(raw_chunks, list):
        return []
    blocks: list[TextBlock] = []
    cursor = 0
    for index, chunk in enumerate(raw_chunks, 1):
        if not isinstance(chunk, dict):
            continue
        content = normalize_text(str(chunk.get("content") or ""))
        if not content:
            continue
        found = doc.text.find(content, cursor)
        start = found if found >= 0 else cursor
        end = start + len(content)
        cursor = max(cursor, end)
        digest = hashlib.sha1(f"{doc.source_file}:deepseekmine:{index}:{content[:64]}".encode("utf-8")).hexdigest()[:12]
        blocks.append(
            TextBlock(
                block_id=digest,
                source_file=doc.source_file,
                file_type=doc.file_type,
                content=content,
                start=start,
                end=end,
                page=str(chunk.get("page") or page_for_span(doc.page_spans, start, end)),
            )
        )
    return blocks


def split_markdown_with_semantic_spans(doc: ParsedDocument, max_chars: int, min_chars: int, overlap: int) -> list[TextBlock]:
    base_blocks = split_markdown_heading_blocks(doc, max_chars, min_chars, overlap)
    if not base_blocks:
        base_blocks = split_paragraph_blocks(doc, max_chars, min_chars, overlap)
    ranges = [{"start": block.start, "end": block.end} for block in base_blocks]
    atomic_ranges = get_ordered_atomic_semantic_ranges(doc)
    if not atomic_ranges:
        return base_blocks

    for atomic in atomic_ranges:
        first = next((index for index, item in enumerate(ranges) if item["end"] > atomic["start"] and item["start"] < atomic["end"]), -1)
        if first < 0:
            insert_at = next((index for index, item in enumerate(ranges) if item["start"] > atomic["start"]), len(ranges))
            ranges.insert(insert_at, atomic)
            continue
        last = first
        while last + 1 < len(ranges) and ranges[last + 1]["start"] < atomic["end"]:
            last += 1
        merged = {
            "start": min(ranges[first]["start"], atomic["start"]),
            "end": max(ranges[last]["end"], atomic["end"]),
        }
        ranges[first:last + 1] = [merged]

    blocks: list[TextBlock] = []
    for item in merge_overlapping_ranges(ranges):
        start, end = item["start"], item["end"]
        content = normalize_text(doc.text[start:end])
        if len(content) < min(min_chars, 20):
            continue
        if len(content) <= max_chars * 2:
            blocks.append(make_block(doc, content, start, end))
            continue
        for part_start, part_end, part in split_long_unit(doc.text, start, end, max_chars, overlap):
            if len(part) >= min(min_chars, 20):
                blocks.append(make_block(doc, part, part_start, part_end))
    return blocks


def get_ordered_atomic_semantic_ranges(doc: ParsedDocument) -> list[dict[str, int]]:
    ranges: list[dict[str, int]] = []
    for span in doc.semantic_spans or []:
        span_type = str(span.get("type") or "").lower()
        if span_type not in {"table", "image"}:
            continue
        try:
            start = max(0, min(len(doc.text), int(span.get("start"))))
            end = max(start, min(len(doc.text), int(span.get("end"))))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        if span_type == "image":
            start = expand_image_start(doc.text, start)
        if span_type == "table":
            start, end = trim_table_trailing_image(doc.text, start, end)
        clamped = clamp_range_to_markdown_section(doc.text, start, end)
        if clamped:
            ranges.append(clamped)
    return sorted(ranges, key=lambda item: (item["start"], item["end"]))


def expand_image_start(text: str, start: int) -> int:
    look_behind_start = max(0, start - 2000)
    look_behind = text[look_behind_start:start]
    match = re.search(r"(?:^|\n)[ \t]*(?:!\[[^\]\r\n]*\]\([^)]+\)|<img\b[\s\S]*?>)[ \t]*(?:\n[ \t]*)*$", look_behind, flags=re.I)
    return look_behind_start + match.start() if match else start


def trim_table_trailing_image(text: str, start: int, end: int) -> tuple[int, int]:
    segment = text[start:end]
    match = re.search(r"(?:\n[ \t]*)+(?:!\[[^\]\r\n]*\]\([^)]+\)|<img\b[\s\S]*?>)[ \t]*(?:\n[ \t]*)*$", segment, flags=re.I)
    if not match:
        return start, end
    before_image = segment[: match.start()]
    if "<table" not in before_image.lower():
        return start, end
    return start, start + match.start()


def clamp_range_to_markdown_section(text: str, start: int, end: int) -> dict[str, int] | None:
    section_start = 0
    section_end = len(text)
    for match in re.finditer(r"(?m)^#{1,6}\s+\S.*$", text):
        if match.start() <= start:
            section_start = match.start()
            continue
        section_end = match.start()
        break
    left = max(section_start, start)
    right = min(section_end, end)
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    if right <= left:
        return None
    return {"start": left, "end": right}


def merge_overlapping_ranges(ranges: list[dict[str, int]]) -> list[dict[str, int]]:
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda item: (item["start"], item["end"]))
    merged = [dict(ordered[0])]
    for item in ordered[1:]:
        last = merged[-1]
        if item["start"] <= last["end"]:
            last["end"] = max(last["end"], item["end"])
        else:
            merged.append(dict(item))
    return merged


def split_markdown_heading_blocks(doc: ParsedDocument, max_chars: int, min_chars: int, overlap: int) -> list[TextBlock]:
    headings = list(re.finditer(r"(?m)^(#{1,6})\s+(.+?)\s*#*\s*$", doc.text))
    if not headings:
        return []

    blocks: list[TextBlock] = []
    markdown_min_chars = min(min_chars, 20)
    if headings[0].start() > 0:
        preface = normalize_text(doc.text[: headings[0].start()])
        if len(preface) >= markdown_min_chars:
            blocks.append(make_block(doc, preface, 0, headings[0].start()))

    stack: list[tuple[int, str]] = []
    for index, match in enumerate(headings):
        level = len(match.group(1))
        title = normalize_text(match.group(2).strip())
        stack = [item for item in stack if item[0] < level]
        stack.append((level, title))
        body_start = match.end()
        body_end = headings[index + 1].start() if index + 1 < len(headings) else len(doc.text)
        body = normalize_text(doc.text[body_start:body_end])
        context = "\n".join(f"{'#' * item_level} {item_title}" for item_level, item_title in stack)
        section = normalize_text(f"{context}\n\n{body}" if body else context)
        if len(section) < markdown_min_chars:
            continue
        if len(section) <= max_chars:
            blocks.append(make_block(doc, section, match.start(), body_end))
            continue

        body_limit = max(320, max_chars - len(context) - 2)
        if body:
            for part_start, part_end, part in split_long_unit(doc.text, body_start, body_end, body_limit, overlap):
                content = normalize_text(f"{context}\n\n{part}")
                if len(content) >= markdown_min_chars:
                    blocks.append(make_block(doc, content, match.start() if part_start == body_start else part_start, part_end))
        elif len(context) >= markdown_min_chars:
            blocks.append(make_block(doc, context, match.start(), match.end()))
    return blocks


def split_paragraph_blocks(doc: ParsedDocument, max_chars: int, min_chars: int, overlap: int) -> list[TextBlock]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.text) if p.strip()]
    units: list[tuple[int, int, str]] = []
    cursor = 0
    for paragraph in paragraphs:
        found = doc.text.find(paragraph, cursor)
        start = found if found >= 0 else cursor
        end = start + len(paragraph)
        cursor = end
        if len(paragraph) <= max_chars:
            units.append((start, end, paragraph))
        else:
            units.extend(split_long_unit(doc.text, start, end, max_chars, overlap))

    blocks: list[TextBlock] = []
    current: list[str] = []
    block_start = 0
    block_end = 0
    for start, end, content in units:
        if current and sum(len(x) for x in current) + len(content) + 2 > max_chars:
            block_text = normalize_text("\n\n".join(current))
            if len(block_text) >= min_chars:
                blocks.append(make_block(doc, block_text, block_start, block_end))
            current = []
        if not current:
            block_start = start
        current.append(content)
        block_end = end
    if current:
        block_text = normalize_text("\n\n".join(current))
        if len(block_text) >= min_chars:
            blocks.append(make_block(doc, block_text, block_start, block_end))
    return blocks


def split_long_unit(text: str, start: int, end: int, max_chars: int, overlap: int) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    pos = start
    while pos < end:
        right = min(end, pos + max_chars)
        if right < end:
            sentence_breaks = [m.end() for m in re.finditer(r"[。！？.!?]\s*", text[pos:right])]
            if sentence_breaks:
                right = pos + sentence_breaks[-1]
        content = normalize_text(text[pos:right])
        if content:
            out.append((pos, right, content))
        if right >= end:
            break
        next_pos = right - max(0, overlap)
        if next_pos <= pos:
            next_pos = right
        if next_pos <= pos:
            next_pos = pos + 1
        pos = min(end, max(start, next_pos))
    return out


def make_block(doc: ParsedDocument, block_text: str, start: int, end: int) -> TextBlock:
    digest = hashlib.sha1(f"{doc.source_file}:{start}:{end}:{block_text[:64]}".encode("utf-8")).hexdigest()[:12]
    return TextBlock(
        block_id=digest,
        source_file=doc.source_file,
        file_type=doc.file_type,
        content=block_text,
        start=start,
        end=end,
        page=page_for_span(doc.page_spans, start, end),
    )


class LLMClient:
    def __init__(self, backend: str, model: str, base_url: str, timeout: int) -> None:
        self.backend = backend
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        if self.backend == "none":
            raise RuntimeError("LLM backend is disabled")
        if self.backend == "ollama":
            return self._ollama(prompt)
        if self.backend == "configured":
            return self._configured_provider(prompt)
        if self.backend == "mimo":
            return self._openai_compatible(
                prompt,
                os.environ.get("MIMO_API_KEY", ""),
                self.base_url or os.environ.get("MIMO_BASE", "https://token-plan-cn.xiaomimimo.com/v1"),
                "api-key",
            )
        if self.backend == "openai":
            return self._openai_compatible(
                prompt,
                os.environ.get("OPENAI_API_KEY", ""),
                self.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                "Authorization",
            )
        if self.backend in {"siliconflow", "deepseek", "qwen", "custom"}:
            return self._preset_provider(prompt)
        raise ValueError(f"unknown backend: {self.backend}")

    def _provider_messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "你是严格的 RAG 测评数据标注员，只输出可解析 JSON。",
            },
            {"role": "user", "content": prompt},
        ]

    def _configured_provider(self, prompt: str) -> str:
        config = load_llm_config()
        default_cli_model = os.environ.get("RAG_BUILDER_MODEL", "qwen3.5:4b")
        if self.model and self.model != default_cli_model:
            config.model = self.model
        if self.base_url:
            config.api_url = self.base_url
        try:
            data = chat_completion(self._provider_messages(prompt), 4096, self.timeout, "dataset generation", config)
        except ProviderQuotaExceededError as exc:
            raise LLMQuotaExceededError(str(exc)) from exc
        return message_text(data)

    def _preset_provider(self, prompt: str) -> str:
        config = default_provider_config(self.backend)
        default_cli_model = os.environ.get("RAG_BUILDER_MODEL", "qwen3.5:4b")
        if self.model and self.model != default_cli_model:
            config.model = self.model
        if self.base_url:
            config.api_url = self.base_url
        try:
            data = chat_completion(self._provider_messages(prompt), 4096, self.timeout, "dataset generation", config)
        except ProviderQuotaExceededError as exc:
            raise LLMQuotaExceededError(str(exc)) from exc
        return message_text(data)

    def _ollama(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.base_url or 'http://127.0.0.1:11434'}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.1, "num_ctx": 32768},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return (resp.json().get("message") or {}).get("content", "")

    def _openai_compatible(self, prompt: str, key: str, base_url: str, auth_header: str) -> str:
        if not key:
            raise RuntimeError(f"{self.backend} API key is missing")
        headers = {"Content-Type": "application/json"}
        if auth_header == "Authorization":
            headers[auth_header] = f"Bearer {key}"
        else:
            headers[auth_header] = key
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是严格的 RAG 测评数据标注员，只输出可解析 JSON。",
                },
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.95,
            "stream": False,
            "stop": None,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        if self.backend == "mimo":
            payload["thinking"] = {"type": "disabled"}
        retry_delays = [2, 6, 12]
        resp = None
        for attempt in range(len(retry_delays) + 1):
            resp = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            body = resp.text[:800].replace("\n", " ")
            if resp.status_code >= 400 and is_quota_error(resp.status_code, body):
                raise LLMQuotaExceededError(f"{self.backend} quota exhausted: HTTP {resp.status_code} {body}")
            if resp.status_code == 429:
                if attempt < len(retry_delays):
                    time.sleep(retry_delays[attempt])
                    continue
                raise LLMQuotaExceededError(f"{self.backend} request limit or Token Plan quota exhausted: HTTP 429 {body}")
            if resp.status_code in {500, 502, 503, 504} and attempt < len(retry_delays):
                time.sleep(retry_delays[attempt])
                continue
            break
        assert resp is not None
        resp.raise_for_status()
        data = resp.json()
        return (data.get("choices", [{}])[0].get("message") or {}).get("content", "")


def build_prompt(block: TextBlock, max_items: int) -> str:
    return f"""
请基于下面证据片段生成最多 {max_items} 条 RAG 测评问答。要求：
1. 问题必须只能由证据片段回答，不要依赖常识或外部知识。
2. 答案必须短、确定，并且答案文本或每个答案要点必须能在证据原文中直接找到。
3. 优先生成事实抽取、数值/日期、列表、条件限定、对比类问题；不要生成“这段话主要讲什么”这种摘要题。
4. evidence_quote 必须原样摘自证据片段，足以支撑答案，但不要超过 500 字。
5. 如果证据不足以形成高质量问题，返回空 items。

只输出 JSON，格式如下：
{{
  "items": [
    {{
      "query": "问题",
      "answer": ["答案1", "答案2"],
      "evidence_quote": "原文证据",
      "qa_type": "fact|numeric|date|list|comparison",
      "difficulty": "easy|medium|hard",
      "answer_aliases": []
    }}
  ]
}}

证据来源：{block.source_file} 页码/位置：{block.page or "unknown"}
证据片段：
<<<
{block.content}
>>>
""".strip()


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _repair_json_candidate(text: str) -> str:
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return re.sub(r'\\(?!["\\/bfnrtu])', lambda _: r"\\", text)


def extract_json_object(text: str) -> dict[str, Any]:
    text = _strip_json_fences(text)
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        if snippet != text:
            candidates.append(snippet)

    last_error = "empty response"
    for candidate in candidates:
        for attempt in (candidate, _repair_json_candidate(candidate)):
            try:
                payload = json.loads(attempt)
            except json.JSONDecodeError as exc:
                last_error = str(exc)
                continue
            if isinstance(payload, dict):
                return payload
            last_error = "top-level JSON is not an object"
    raise ValueError(f"model did not return a valid JSON object: {last_error}")


def validate_generated_item(raw: dict[str, Any], block: TextBlock) -> tuple[dict[str, Any] | None, str]:
    query = normalize_text(str(raw.get("query", "")))
    answers = raw.get("answer", [])
    if isinstance(answers, str):
        answers = [answers]
    answers = [normalize_text(str(a)) for a in answers if normalize_text(str(a))]
    quote = normalize_text(str(raw.get("evidence_quote", "")))
    if not query or not answers or not quote:
        return None, "missing query/answer/evidence_quote"
    if normalize_for_match(quote) not in normalize_for_match(block.content):
        return None, "evidence quote is not copied from source chunk"
    quote_norm = normalize_for_match(quote)
    missing_answers = [a for a in answers if normalize_for_match(a) not in quote_norm]
    if missing_answers:
        return None, f"answers not found in evidence quote: {missing_answers}"
    return {
        "query": query,
        "answer": answers,
        "evidence_quote": quote,
        "qa_type": str(raw.get("qa_type") or "fact"),
        "difficulty": str(raw.get("difficulty") or "medium"),
        "answer_aliases": raw.get("answer_aliases") if isinstance(raw.get("answer_aliases"), list) else [],
    }, ""


def heuristic_items(block: TextBlock, max_items: int) -> list[dict[str, Any]]:
    patterns = [
        r"([一二三四五六七八九十百千万亿\d,.]+(?:\.\d+)?%?)",
        r"(\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日|\d{4}年)",
    ]
    sentences = [s.strip() for s in re.split(r"(?<=[。！？.!?])\s*", block.content) if 30 <= len(s.strip()) <= 260]
    out: list[dict[str, Any]] = []
    for sentence in sentences:
        for pattern in patterns:
            match = re.search(pattern, sentence)
            if not match:
                continue
            answer = match.group(1)
            out.append(
                {
                    "query": f"根据材料，'{sentence[:30]}' 相关表述中的关键数值或时间是什么？",
                    "answer": [answer],
                    "evidence_quote": sentence,
                    "qa_type": "numeric" if re.search(r"\d|%", answer) else "fact",
                    "difficulty": "easy",
                    "answer_aliases": [],
                }
            )
            break
        if len(out) >= max_items:
            break
    return out


def choose_negatives(block: TextBlock, blocks: list[TextBlock], answers: list[str], count: int) -> list[str]:
    if count <= 0:
        return []
    answer_norms = [normalize_for_match(a) for a in answers]
    query_terms = set(re.findall(r"[\w\u4e00-\u9fff]{2,}", block.content[:300].lower()))
    candidates: list[tuple[float, TextBlock]] = []
    for candidate in blocks:
        if candidate.block_id == block.block_id:
            continue
        cand_norm = normalize_for_match(candidate.content)
        if any(a and a in cand_norm for a in answer_norms):
            continue
        cand_terms = set(re.findall(r"[\w\u4e00-\u9fff]{2,}", candidate.content[:300].lower()))
        overlap = len(query_terms & cand_terms) / max(1, len(query_terms | cand_terms))
        candidates.append((overlap, candidate))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [c.content for _, c in candidates[: count * 2]]
    random.shuffle(selected)
    return selected[:count]


def emit_progress(args: argparse.Namespace, event: str, **payload: Any) -> None:
    callback: Callable[..., None] | None = getattr(args, "progress_callback", None)
    if callback is None:
        return
    callback(event=event, **payload)


def distribute_weighted(weights: list[int], total: int) -> list[int]:
    if total <= 0 or not weights:
        return [0 for _ in weights]
    quotas = [0 for _ in weights]
    indexed_weights = [(index, max(1, weight)) for index, weight in enumerate(weights)]
    if total < len(weights):
        for index, _ in sorted(indexed_weights, key=lambda item: item[1], reverse=True)[:total]:
            quotas[index] = 1
        return quotas

    quotas = [1 for _ in weights]
    remaining = total - len(weights)
    if remaining <= 0:
        return quotas

    weight_sum = sum(weight for _, weight in indexed_weights)
    exact = [remaining * weight / weight_sum for _, weight in indexed_weights]
    floors = [int(value) for value in exact]
    for index, floor_value in enumerate(floors):
        quotas[index] += floor_value
    leftover = remaining - sum(floors)
    remainders = sorted(
        [(index, exact[index] - floors[index], indexed_weights[index][1]) for index in range(len(weights))],
        key=lambda item: (item[1], item[2]),
        reverse=True,
    )
    for index, _, _ in remainders[:leftover]:
        quotas[index] += 1
    return quotas


def allocate_block_quotas(blocks: list[TextBlock], target_count: int, max_per_source: int) -> list[int]:
    if not blocks or target_count <= 0:
        return [0 for _ in blocks]

    source_to_indices: dict[str, list[int]] = {}
    for index, block in enumerate(blocks):
        source_to_indices.setdefault(block.source_file, []).append(index)

    source_names = list(source_to_indices)
    source_weights = [sum(len(blocks[index].content) for index in source_to_indices[source]) for source in source_names]
    effective_source_cap = max(1, max_per_source)
    reserve_total = max(10, (target_count + 1) // 2)
    desired_total = target_count + reserve_total
    per_source_attempt_cap = effective_source_cap + max(5, (effective_source_cap + 1) // 2)
    total_attempt_cap = per_source_attempt_cap * len(source_names)
    desired_total = min(desired_total, total_attempt_cap)
    source_quotas = distribute_weighted(source_weights, desired_total)

    quotas = [0 for _ in blocks]
    for source, source_quota in zip(source_names, source_quotas):
        source_quota = min(source_quota, per_source_attempt_cap)
        indices = source_to_indices[source]
        block_weights = [len(blocks[index].content) for index in indices]
        block_quotas = distribute_weighted(block_weights, source_quota)
        for block_index, quota in zip(indices, block_quotas):
            quotas[block_index] = quota
    return quotas


def make_row(
    row_id: int,
    item: dict[str, Any],
    block: TextBlock,
    blocks: list[TextBlock],
    negative_count: int,
    generator: str,
) -> dict[str, Any]:
    quote = item["evidence_quote"]
    quote_start = block.content.find(quote)
    absolute_start = block.start + max(0, quote_start)
    absolute_end = absolute_start + len(quote)
    evidence = Evidence(
        source_file=block.source_file,
        file_type=block.file_type,
        chunk_id=block.block_id,
        page=block.page,
        quote=quote,
        char_start=absolute_start,
        char_end=absolute_end,
    )
    answers = item["answer"]
    row: dict[str, Any] = {
        "id": row_id,
        "query": item["query"],
        "answer": answers,
        "positive": [quote],
        "negative": choose_negatives(block, blocks, answers, negative_count),
        "evidence": [asdict(evidence)],
        "qa_type": item["qa_type"],
        "difficulty": item["difficulty"],
        "answer_aliases": item.get("answer_aliases", []),
        "meta": {
            "generator": generator,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "schema": "rag-eval-builder-v1",
            "source_chunk_id": block.block_id,
        },
    }
    if len(answers) >= 2:
        row["answer1"] = [answers[0]]
        row["answer2"] = [answers[1]]
        row["asnwer1"] = [answers[0]]  # legacy typo kept for old zh_int-style consumers.
    return row


def build_dataset(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    random.seed(args.seed)
    skipped: list[dict[str, str]] = []
    parsed_docs: list[ParsedDocument] = []
    emit_progress(args, "parse_start", input_count=len(args.input))
    for source, payload in iter_input_payloads([Path(p) for p in args.input], args.max_zip_depth):
        try:
            doc = parse_bytes(payload, source)
            if len(doc.text) < args.min_doc_chars:
                skipped.append({"source": source, "reason": f"too little extracted text ({len(doc.text)} chars)"})
                emit_progress(args, "parse_skip", source=source, reason=skipped[-1]["reason"])
                continue
            parsed_docs.append(doc)
            emit_progress(args, "parse_doc", source=source, chars=len(doc.text), parser=doc.parser)
        except Exception as exc:
            skipped.append({"source": source, "reason": str(exc)})
            emit_progress(args, "parse_skip", source=source, reason=str(exc))

    blocks: list[TextBlock] = []
    markdown_document_count = 0
    mineru_semantic_document_count = 0
    for doc in parsed_docs:
        if is_markdown_document(doc):
            markdown_document_count += 1
        if doc.semantic_spans:
            mineru_semantic_document_count += 1
        blocks.extend(split_blocks(doc, args.chunk_chars, args.min_chunk_chars, args.chunk_overlap))
    if args.shuffle_chunks:
        random.shuffle(blocks)
    emit_progress(
        args,
        "chunk_plan",
        document_count=len(parsed_docs),
        block_count=len(blocks),
        markdown_document_count=markdown_document_count,
        mineru_semantic_document_count=mineru_semantic_document_count,
        splitter="mineru_semantic" if mineru_semantic_document_count else "markdown_heading" if markdown_document_count else "paragraph_window",
    )

    llm = LLMClient(args.backend, args.model, args.base_url, args.timeout)
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    source_counts: dict[str, int] = {}
    stopped_reason = ""
    block_quotas = allocate_block_quotas(blocks, args.target_count, args.max_per_source)
    planned_total = sum(block_quotas)
    emit_progress(
        args,
        "qa_plan",
        planned_total=planned_total,
        target_count=args.target_count,
        max_per_block=max(block_quotas) if block_quotas else 0,
        active_blocks=sum(1 for quota in block_quotas if quota > 0),
        block_count=len(blocks),
    )

    for block_index, block in enumerate(blocks, 1):
        if len(rows) >= args.target_count:
            break
        planned_items = block_quotas[block_index - 1] if block_index - 1 < len(block_quotas) else 0
        source_remaining = args.max_per_source - source_counts.get(block.source_file, 0)
        target_remaining = args.target_count - len(rows)
        planned_items = min(planned_items, source_remaining, target_remaining)
        if planned_items <= 0:
            continue
        if source_counts.get(block.source_file, 0) >= args.max_per_source:
            emit_progress(
                args,
                "qa_chunk_skip",
                block_index=block_index,
                total_blocks=len(blocks),
                row_count=len(rows),
                target_count=args.target_count,
                source=block.source_file,
                chunk_id=block.block_id,
                page=block.page,
                reason="source max reached",
            )
            continue
        raw_items: list[dict[str, Any]]
        generator = args.backend
        emit_progress(
            args,
            "qa_chunk_start",
            block_index=block_index,
            total_blocks=len(blocks),
            row_count=len(rows),
            target_count=args.target_count,
            source=block.source_file,
            chunk_id=block.block_id,
            page=block.page,
            chars=len(block.content),
            planned_items=planned_items,
        )
        try:
            response = llm.generate(build_prompt(block, planned_items))
            payload = extract_json_object(response)
            raw_items = payload.get("items", []) if isinstance(payload.get("items"), list) else []
        except LLMQuotaExceededError as exc:
            stopped_reason = str(exc)
            rejected.append(
                {
                    "source": block.source_file,
                    "chunk_id": block.block_id,
                    "reason": f"api_quota_exhausted: {exc}",
                }
            )
            emit_progress(
                args,
                "api_quota_exhausted",
                block_index=block_index,
                total_blocks=len(blocks),
                row_count=len(rows),
                target_count=args.target_count,
                source=block.source_file,
                chunk_id=block.block_id,
                page=block.page,
                reason=stopped_reason,
            )
            break
        except Exception as exc:
            if not args.allow_heuristic:
                rejected.append(
                    {
                        "source": block.source_file,
                        "chunk_id": block.block_id,
                        "reason": f"generation_parse_error: {exc}",
                    }
                )
                raw_items = []
            else:
                raw_items = heuristic_items(block, planned_items)
                generator = "heuristic"

        for raw in raw_items:
            item, reason = validate_generated_item(raw, block)
            if item is None:
                rejected.append({"source": block.source_file, "chunk_id": block.block_id, "reason": reason})
                continue
            rows.append(make_row(len(rows), item, block, blocks, args.negative_count, generator))
            source_counts[block.source_file] = source_counts.get(block.source_file, 0) + 1
            if len(rows) >= args.target_count or source_counts[block.source_file] >= args.max_per_source:
                break
        emit_progress(
            args,
            "qa_chunk_done",
            block_index=block_index,
            total_blocks=len(blocks),
            row_count=len(rows),
            target_count=args.target_count,
            source=block.source_file,
            chunk_id=block.block_id,
            page=block.page,
            accepted=len(rows),
            rejected_count=len(rejected),
            raw_count=len(raw_items),
            planned_items=planned_items,
        )

    report = {
        "input": args.input,
        "output": str(Path(args.output).resolve()),
        "parsed_documents": [asdict(d) | {"text": f"{len(d.text)} chars"} for d in parsed_docs],
        "block_count": len(blocks),
        "block_quota_count": planned_total,
        "max_questions_per_block": max(block_quotas) if block_quotas else 0,
        "active_question_blocks": sum(1 for quota in block_quotas if quota > 0),
        "markdown_document_count": markdown_document_count,
        "splitter": "markdown_heading" if markdown_document_count else "paragraph_window",
        "row_count": len(rows),
        "skipped": skipped,
        "rejected": rejected[:200],
        "rejected_count": len(rejected),
        "stopped_reason": stopped_reason,
        "completed": not stopped_reason,
        "supported_extensions": sorted(SUPPORTED_UPLOAD_EXTENSIONS),
        "notes": [
            "Images require OCR before this standalone builder can generate QA.",
            "Generated rows should be manually spot-checked before becoming benchmark truth.",
        ],
    }
    return rows, report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RAG QA evaluation datasets from documents.")
    parser.add_argument("--input", nargs="+", required=True, help="Input files, directories, or .zip archives.")
    parser.add_argument("--output", required=True, help="Output JSONL path, e.g. E:\\RAG\\data\\my_eval.json.")
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument("--items-per-chunk", type=int, default=1)
    parser.add_argument("--max-per-source", type=int, default=100)
    parser.add_argument("--negative-count", type=int, default=0)
    parser.add_argument("--chunk-chars", type=int, default=1400)
    parser.add_argument("--min-chunk-chars", type=int, default=180)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--min-doc-chars", type=int, default=200)
    parser.add_argument("--backend", choices=["ollama", "mimo", "siliconflow", "deepseek", "qwen", "custom", "configured", "openai", "none"], default="ollama")
    parser.add_argument("--model", default=os.environ.get("RAG_BUILDER_MODEL", "qwen3.5:4b"))
    parser.add_argument("--base-url", default=os.environ.get("RAG_BUILDER_BASE_URL", ""))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--allow-heuristic", action="store_true", help="Use simple extraction fallback if LLM fails.")
    parser.add_argument("--shuffle-chunks", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-zip-depth", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    rows, report = build_dataset(args)
    output = Path(args.output)
    write_jsonl(output, rows)
    report_path = output.with_suffix(output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[builder] wrote {len(rows)} rows -> {output.resolve()}")
    print(f"[builder] report -> {report_path.resolve()}")
    if report["skipped"]:
        print(f"[builder] skipped files: {len(report['skipped'])}")
    if not rows:
        print("[builder] no rows generated; inspect the report for parser/generation failures")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
