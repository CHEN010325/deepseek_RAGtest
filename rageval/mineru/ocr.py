"""
MinerU OCR/parse integration for the RAG dataset builder.

This module follows the precise parsing API documented at:
https://mineru.net/apiManage/docs

For PDFs over 200 pages it physically splits the file into <=200-page PDF
parts before submitting them to MinerU, mirroring the strategy used in
E:\\deepseekmine's MinerU batch processor.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import requests


MINERU_API_BASE = "https://mineru.net/api/v4"
MAX_PAGES_PER_TASK = 200
MAX_FILES_PER_BATCH = 50
MAX_UPLOAD_BATCH_BYTES = 256 * 1024 * 1024
POLL_INTERVAL_SECONDS = 3
POLL_TIMEOUT_SECONDS = 30 * 60

MINERU_SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".png",
    ".jpg",
    ".jpeg",
    ".jp2",
    ".webp",
    ".gif",
    ".bmp",
    ".html",
    ".htm",
}


@dataclass
class MineruOptions:
    token: str
    api_base: str = MINERU_API_BASE
    model_version: str = "vlm"
    html_model_version: str = "MinerU-HTML"
    language: str = "ch"
    is_ocr: bool = True
    enable_formula: bool = True
    enable_table: bool = True
    timeout: int = 120


@dataclass
class UploadPart:
    original_path: str
    upload_path: str
    upload_name: str
    data_id: str
    part_index: int
    part_total: int
    page_start: int
    page_end: int
    page_offset: int
    size_bytes: int


@dataclass
class MineruParsedFile:
    original_path: str
    success: bool
    markdown_path: str | None = None
    metadata_path: str | None = None
    zip_paths: list[str] | None = None
    json_paths: list[str] | None = None
    content_list_paths: list[str] | None = None
    middle_json_paths: list[str] | None = None
    combined_json_path: str | None = None
    text_chars: int = 0
    part_count: int = 0
    error: str | None = None
    parts: list[dict] | None = None


@dataclass
class MineruPartArtifact:
    markdown: str
    markdown_name: str
    zip_path: str
    json_paths: list[str]
    content_list_path: str | None = None
    middle_json_path: str | None = None
    page_offset: int = 0


ProgressCallback = Callable[[str], None]


class MineruError(RuntimeError):
    pass


class MineruQuotaExceededError(MineruError):
    pass


QUOTA_ERROR_PATTERNS = [
    "insufficient balance",
    "insufficient_quota",
    "quota exceeded",
    "quota_exceeded",
    "quota exhausted",
    "out of credits",
    "billing",
    "payment required",
    "resource exhausted",
    "request limit",
    "余额不足",
    "额度不足",
    "配额不足",
    "账户余额",
    "资源耗尽",
    "欠费",
]


def raise_for_api_status(response: requests.Response, provider: str) -> None:
    if response.status_code < 400:
        return
    body = response.text[:800].replace("\n", " ")
    text = body.lower()
    if response.status_code in {402, 429} or any(pattern in text for pattern in QUOTA_ERROR_PATTERNS):
        raise MineruQuotaExceededError(f"{provider} quota or request limit exhausted: HTTP {response.status_code} {body}")
    response.raise_for_status()


class MineruClient:
    def __init__(self, options: MineruOptions) -> None:
        self.options = options
        self.base = options.api_base.rstrip("/")
        if not options.token.strip():
            raise MineruError("MinerU token is required")

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.options.token.strip()}",
            "Content-Type": "application/json",
            "Accept": "*/*",
        }

    def create_batch_upload_urls(self, parts: list[UploadPart], model_version: str) -> dict:
        payload = {
            "files": [
                {
                    "name": safe_mineru_filename(part.upload_name),
                    "data_id": part.data_id,
                    "is_ocr": self.options.is_ocr,
                }
                for part in parts
            ],
            "model_version": model_version,
            "language": self.options.language,
            "enable_formula": self.options.enable_formula,
            "enable_table": self.options.enable_table,
        }
        response = requests.post(
            f"{self.base}/file-urls/batch",
            headers=self.headers,
            json=payload,
            timeout=self.options.timeout,
        )
        raise_for_api_status(response, "MinerU upload URL request")
        data = response.json()
        if data.get("code") != 0:
            message = str(data.get("msg") or data)
            if any(pattern in message.lower() for pattern in QUOTA_ERROR_PATTERNS):
                raise MineruQuotaExceededError(f"MinerU upload URL request failed: {message}")
            raise MineruError(f"MinerU upload URL request failed: {message}")
        urls = data.get("data", {}).get("file_urls") or []
        if len(urls) != len(parts):
            raise MineruError(f"MinerU returned {len(urls)} upload URLs for {len(parts)} files")
        return data

    def upload_file(self, upload_url: str, file_path: str) -> None:
        with open(file_path, "rb") as f:
            response = requests.put(upload_url, data=f, timeout=max(self.options.timeout, 300))
        if response.status_code not in (200, 201, 204):
            raise_for_api_status(response, "MinerU file upload")
            raise MineruError(f"MinerU file upload failed: HTTP {response.status_code} {response.text[:200]}")

    def get_batch_status(self, batch_id: str) -> dict:
        response = requests.get(
            f"{self.base}/extract-results/batch/{batch_id}",
            headers=self.headers,
            timeout=self.options.timeout,
        )
        raise_for_api_status(response, "MinerU batch status")
        data = response.json()
        if data.get("code") != 0:
            message = str(data.get("msg") or data)
            if any(pattern in message.lower() for pattern in QUOTA_ERROR_PATTERNS):
                raise MineruQuotaExceededError(f"MinerU batch status failed: {message}")
            raise MineruError(f"MinerU batch status failed: {message}")
        return data

    def download_artifacts(self, zip_url: str, part: UploadPart, output_dir: Path) -> MineruPartArtifact:
        response = requests.get(zip_url, timeout=max(self.options.timeout, 300))
        raise_for_api_status(response, "MinerU result download")
        zip_dir = output_dir / "mineru_zips"
        json_dir = output_dir / "mineru_json" / sanitize_artifact_stem(part.upload_name)
        zip_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        saved_zip = unique_path(zip_dir / f"{sanitize_artifact_stem(part.upload_name)}.zip")
        saved_zip.write_bytes(response.content)

        with zipfile.ZipFile(io_bytes(response.content)) as zf:
            md_names = [n for n in zf.namelist() if n.lower().endswith((".md", ".markdown"))]
            if not md_names:
                raise MineruError("MinerU result zip contains no Markdown file")
            preferred = next((n for n in md_names if Path(n).name.lower() == "full.md"), md_names[0])
            markdown = zf.read(preferred).decode("utf-8", errors="replace").strip()

            json_paths: list[str] = []
            content_list_path: str | None = None
            middle_json_path: str | None = None
            json_names = [n for n in zf.namelist() if n.lower().endswith(".json") and not n.endswith("/")]
            for name in json_names:
                target = unique_path(json_dir / sanitize_zip_member_name(name))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(name))
                json_paths.append(str(target))
                lower_name = name.lower()
                if re.search(r"_content_list\.json$", lower_name):
                    content_list_path = str(target)
                elif middle_json_path is None and (re.search(r"_middle\.json$", lower_name) or re.search(r"middle.*\.json$", lower_name)):
                    middle_json_path = str(target)
            if middle_json_path is None:
                for path in json_paths:
                    if not re.search(r"_content_list\.json$", Path(path).name.lower()):
                        middle_json_path = path
                        break

        return MineruPartArtifact(
            markdown=markdown,
            markdown_name=preferred,
            zip_path=str(saved_zip),
            json_paths=json_paths,
            content_list_path=content_list_path,
            middle_json_path=middle_json_path,
            page_offset=part.page_offset,
        )

    def download_markdown(self, zip_url: str) -> str:
        response = requests.get(zip_url, timeout=max(self.options.timeout, 300))
        response.raise_for_status()
        with zipfile.ZipFile(io_bytes(response.content)) as zf:
            md_names = [n for n in zf.namelist() if n.lower().endswith((".md", ".markdown"))]
            if not md_names:
                raise MineruError("MinerU result zip contains no Markdown file")
            preferred = next((n for n in md_names if Path(n).name.lower() == "full.md"), md_names[0])
            return zf.read(preferred).decode("utf-8", errors="replace").strip()


def io_bytes(data: bytes):
    import io

    return io.BytesIO(data)


def safe_mineru_filename(filename: str) -> str:
    if not re.search(r"[^\x00-\x7F]", filename):
        return filename
    ext = Path(filename).suffix
    safe_ext = ext if re.match(r"^[A-Za-z0-9.]+$", ext) else ".bin"
    encoded = base64.b64encode(filename.encode("utf-8")).decode("ascii")
    encoded = re.sub(r"[/+=]", "_", encoded)[:200]
    return f"{encoded}{safe_ext}"


def safe_data_id(value: str) -> str:
    digest = hashlib.md5(f"{value}:{time.time_ns()}".encode("utf-8")).hexdigest()
    return digest[:120]


def sanitize_artifact_stem(value: str) -> str:
    stem = Path(value).stem or "mineru_result"
    stem = re.sub(r"[\x00-\x1f<>:\"/\\|?*\s]+", "_", stem).strip("._")
    return stem[:120] or "mineru_result"


def sanitize_zip_member_name(value: str) -> str:
    parts = [p for p in value.replace("\\", "/").split("/") if p and p not in {".", ".."}]
    name = "__".join(parts) if parts else "artifact.json"
    name = re.sub(r"[\x00-\x1f<>:\"/\\|?*]+", "_", name).strip("._")
    return name[:180] or "artifact.json"


def strip_html_tags(value: str) -> str:
    value = re.sub(r"<style[\s\S]*?</style>", " ", value, flags=re.I)
    value = re.sub(r"<script[\s\S]*?</script>", " ", value, flags=re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    return (
        value.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )


def as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item or "").strip() for item in value if str(item or "").strip()]
    text = str(value or "").strip()
    return [text] if text else []


def clean_mineru_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_search_text(value: str) -> dict[str, list[int] | str]:
    text = ""
    index_map: list[int] = []
    prev_was_whitespace = False
    for index, char in enumerate(value):
        if char.isspace():
            if not prev_was_whitespace:
                text += " "
                index_map.append(index)
                prev_was_whitespace = True
            continue
        text += char
        index_map.append(index)
        prev_was_whitespace = False
    return {"text": text, "index_map": index_map}


def normalize_mineru_block_type(value: Any) -> str:
    text = str(value or "").lower()
    if "table" in text:
        return "table"
    if "title" in text:
        return "heading"
    if "image" in text or "figure" in text:
        return "image"
    if "equation" in text:
        return "equation"
    if "list" in text:
        return "list"
    return text or "text"


def get_content_entry_texts(entry: Any) -> list[str]:
    if not isinstance(entry, dict):
        return []
    entry_type = str(entry.get("type") or "").lower()
    texts: list[str] = []
    for key in ("text", "content"):
        if isinstance(entry.get(key), str) and entry[key].strip():
            texts.append(entry[key].strip())
    if entry_type == "image":
        texts.extend(as_string_list(entry.get("image_caption")))
        texts.extend(as_string_list(entry.get("image_footnote")))
    elif entry_type == "table":
        texts.extend(as_string_list(entry.get("table_caption")))
        if isinstance(entry.get("table_body"), str) and entry["table_body"].strip():
            texts.append(strip_html_tags(entry["table_body"]))
        texts.extend(as_string_list(entry.get("table_footnote")))
    elif entry_type == "list":
        texts.extend(as_string_list(entry.get("list_items")))
    elif entry_type == "code":
        texts.extend(as_string_list(entry.get("code_caption")))
        if isinstance(entry.get("code_body"), str) and entry["code_body"].strip():
            texts.append(entry["code_body"])
    return [clean_mineru_text(text) for text in texts if len(clean_mineru_text(text)) >= 8]


def collect_texts_from_mineru_node(node: Any) -> list[str]:
    if not isinstance(node, dict):
        return []
    texts: list[str] = []
    for key in ("content", "text"):
        if isinstance(node.get(key), str):
            texts.append(clean_mineru_text(node[key]))
    if isinstance(node.get("html"), str):
        texts.append(clean_mineru_text(strip_html_tags(node["html"])))
    for key in ("lines", "spans"):
        if isinstance(node.get(key), list):
            for child in node[key]:
                texts.extend(collect_texts_from_mineru_node(child))
    out: list[str] = []
    for text in texts:
        if text and text not in out:
            out.append(text)
    return out


def build_semantic_blocks_from_content_list(content_list: Any) -> list[dict[str, Any]]:
    if not isinstance(content_list, list):
        return []
    blocks: list[dict[str, Any]] = []
    for entry in content_list:
        if not isinstance(entry, dict):
            continue
        block_type = normalize_mineru_block_type(entry.get("type"))
        page_idx = try_int(entry.get("page_idx"))
        title = None
        if block_type == "table":
            title = next(iter(as_string_list(entry.get("table_caption"))), None)
        elif block_type == "image":
            title = next(iter(as_string_list(entry.get("image_caption"))), None)
        texts = get_content_entry_texts(entry)
        if texts:
            blocks.append(
                {
                    "type": block_type,
                    "page": page_idx + 1 if page_idx is not None and page_idx >= 0 else None,
                    "texts": texts,
                    "title": title,
                }
            )
    return blocks


def build_semantic_blocks_from_middle_json(middle_json: Any) -> list[dict[str, Any]]:
    pages = middle_json.get("pdf_info") if isinstance(middle_json, dict) else None
    if not isinstance(pages, list):
        return []
    blocks: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_idx = try_int(page.get("page_idx"))
        page_number = page_idx + 1 if page_idx is not None and page_idx >= 0 else None
        para_blocks = page.get("para_blocks") if isinstance(page.get("para_blocks"), list) else page.get("preproc_blocks")
        if not isinstance(para_blocks, list):
            continue
        for block in para_blocks:
            if not isinstance(block, dict):
                continue
            block_type = normalize_mineru_block_type(block.get("type"))
            if block_type in {"table", "image"} and isinstance(block.get("blocks"), list):
                child_texts: list[str] = []
                title = None
                for child in block["blocks"]:
                    if not isinstance(child, dict):
                        continue
                    child_type = normalize_mineru_block_type(child.get("type"))
                    texts = collect_texts_from_mineru_node(child)
                    if child_type in {block_type, "text"}:
                        child_texts.extend(texts)
                    if title is None and "caption" in str(child.get("type") or "").lower() and texts:
                        title = texts[0]
                unique = [text for index, text in enumerate(child_texts) if text and child_texts.index(text) == index]
                if unique:
                    blocks.append({"type": block_type, "page": page_number, "texts": unique, "title": title})
                continue
            texts = collect_texts_from_mineru_node(block)
            if texts:
                blocks.append(
                    {
                        "type": block_type,
                        "page": page_number,
                        "texts": texts,
                        "title": texts[0] if block_type == "heading" else None,
                    }
                )
    return blocks


def unique_semantic_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for block in blocks:
        key = f"{block.get('type')}:{block.get('page') or ''}:{'|'.join((block.get('texts') or [])[:3])}"
        if key in seen:
            continue
        seen.add(key)
        out.append(block)
    return out


def build_semantic_spans_from_blocks(markdown: str, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not markdown or not blocks:
        return []
    normalized = normalize_search_text(markdown)
    normalized_text = str(normalized["text"])
    index_map = normalized["index_map"]
    assert isinstance(index_map, list)
    located: list[tuple[dict[str, Any], int]] = []
    cursor = 0
    for block in blocks:
        found = -1
        candidates = []
        for text in block.get("texts") or []:
            candidate = str(normalize_search_text(text)["text"]).strip()
            if len(candidate) >= 4 and candidate not in candidates:
                candidates.append(candidate)
        for candidate in candidates[:6]:
            snippets = []
            for length in (160, 120, 80, 50, 30):
                snippet = candidate[: min(length, len(candidate))].strip()
                if len(snippet) >= 4 and snippet not in snippets:
                    snippets.append(snippet)
            for snippet in snippets:
                idx = normalized_text.find(snippet, cursor)
                if idx >= 0:
                    found = idx
                    break
            if found >= 0:
                break
        if found >= 0:
            located.append((block, found))
            cursor = found + 1
    if not located:
        return []

    def to_source_index(value: int | None, fallback: int) -> int:
        if value is None or value <= 0:
            return fallback
        if value >= len(index_map):
            return len(markdown)
        mapped = index_map[value]
        return mapped if isinstance(mapped, int) else fallback

    spans: list[dict[str, Any]] = []
    previous_end = 0
    for index, (block, start_norm) in enumerate(located):
        next_norm = located[index + 1][1] if index + 1 < len(located) else None
        start = max(previous_end, to_source_index(start_norm, previous_end))
        end = max(start, to_source_index(next_norm, len(markdown)))
        if end > start:
            span = {
                "type": block.get("type") or "text",
                "start": start,
                "end": end,
            }
            if block.get("page"):
                span["page"] = block["page"]
            if block.get("title"):
                span["title"] = block["title"]
            spans.append(span)
            previous_end = end
    return spans


def build_page_spans_from_semantic_spans(semantic_spans: list[dict[str, Any]]) -> list[dict[str, int]]:
    by_page: dict[int, dict[str, int]] = {}
    for span in semantic_spans:
        page = try_int(span.get("page"))
        start = try_int(span.get("start"))
        end = try_int(span.get("end"))
        if page is None or start is None or end is None or end <= start:
            continue
        if page in by_page:
            by_page[page]["start"] = min(by_page[page]["start"], start)
            by_page[page]["end"] = max(by_page[page]["end"], end)
        else:
            by_page[page] = {"page": page, "start": start, "end": end}
    return [by_page[page] for page in sorted(by_page)]


def build_page_spans_from_content_list(markdown: str, content_list: Any) -> list[dict[str, int]]:
    if not markdown or not isinstance(content_list, list):
        return []
    page_candidates: dict[int, list[str]] = {}
    for entry in content_list:
        if not isinstance(entry, dict):
            continue
        page_idx = try_int(entry.get("page_idx"))
        if page_idx is None or page_idx < 0:
            continue
        texts = get_content_entry_texts(entry)
        if texts:
            page_candidates.setdefault(page_idx + 1, []).extend(texts)
    ordered_pages = sorted(page_candidates)
    if not ordered_pages:
        return []
    normalized = normalize_search_text(markdown)
    normalized_text = str(normalized["text"])
    index_map = normalized["index_map"]
    assert isinstance(index_map, list)
    located_starts: dict[int, int] = {}
    cursor = 0
    for page in ordered_pages:
        found = -1
        for candidate in page_candidates.get(page, [])[:8]:
            normalized_candidate = str(normalize_search_text(candidate)["text"]).strip()
            if len(normalized_candidate) < 8:
                continue
            snippets = []
            for length in (120, 80, 50, 30):
                snippet = normalized_candidate[: min(length, len(normalized_candidate))].strip()
                if len(snippet) >= 8 and snippet not in snippets:
                    snippets.append(snippet)
            for snippet in snippets:
                idx = normalized_text.find(snippet, cursor)
                if idx >= 0:
                    found = idx
                    break
            if found >= 0:
                break
        if found >= 0:
            located_starts[page] = found
            cursor = found

    def to_source_index(value: int | None, fallback: int) -> int:
        if value is None or value <= 0:
            return fallback
        if value >= len(index_map):
            return len(markdown)
        mapped = index_map[value]
        return mapped if isinstance(mapped, int) else fallback

    spans: list[dict[str, int]] = []
    previous_end = 0
    for index, page in enumerate(ordered_pages):
        start_norm = located_starts.get(page)
        next_norm = None
        for next_page in ordered_pages[index + 1:]:
            if next_page in located_starts:
                next_norm = located_starts[next_page]
                break
        start = max(previous_end, to_source_index(start_norm, 0 if index == 0 else previous_end))
        end = max(start, to_source_index(next_norm, len(markdown)))
        if end > start:
            spans.append({"page": page, "start": start, "end": end})
            previous_end = end
    if not spans and markdown.strip():
        return [{"page": 1, "start": 0, "end": len(markdown)}]
    return spans


def build_mineru_metadata_from_json(markdown: str, mineru_json: Any) -> dict[str, Any]:
    content_list = mineru_json if isinstance(mineru_json, list) else mineru_json.get("content_list", []) if isinstance(mineru_json, dict) else []
    middle_json = mineru_json if isinstance(mineru_json, dict) and isinstance(mineru_json.get("pdf_info"), list) else None
    semantic_blocks = unique_semantic_blocks(
        build_semantic_blocks_from_middle_json(middle_json) + build_semantic_blocks_from_content_list(content_list)
    )
    semantic_spans = build_semantic_spans_from_blocks(markdown, semantic_blocks)
    page_spans = build_page_spans_from_content_list(markdown, content_list)
    semantic_page_spans = build_page_spans_from_semantic_spans(semantic_spans)
    if len({span.get("page") for span in page_spans}) < len({span.get("page") for span in semantic_page_spans}):
        page_spans = semantic_page_spans
    return {
        "page_spans": page_spans or semantic_page_spans,
        "semantic_spans": semantic_spans,
    }


def try_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def read_json_file(path: str | None) -> Any:
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def adjust_content_list_pages(content_list: Any, page_offset: int) -> list[Any]:
    if not isinstance(content_list, list):
        return []
    adjusted: list[Any] = []
    for item in content_list:
        if not isinstance(item, dict):
            continue
        copied = json.loads(json.dumps(item, ensure_ascii=False))
        page_idx = try_int(copied.get("page_idx"))
        if page_idx is not None:
            copied["page_idx"] = page_idx + page_offset
        adjusted.append(copied)
    return adjusted


def adjust_middle_json_pages(middle_json: Any, page_offset: int) -> dict[str, Any] | None:
    if not isinstance(middle_json, dict):
        return None
    copied = json.loads(json.dumps(middle_json, ensure_ascii=False))
    pages = copied.get("pdf_info")
    if isinstance(pages, list):
        for page in pages:
            if isinstance(page, dict):
                page_idx = try_int(page.get("page_idx"))
                if page_idx is not None:
                    page["page_idx"] = page_idx + page_offset
    return copied


def build_combined_mineru_json(success_parts: list[UploadPart], artifacts: dict[str, MineruPartArtifact]) -> dict[str, Any]:
    content_list: list[Any] = []
    pdf_info: list[Any] = []
    source_parts: list[dict[str, Any]] = []
    for part in success_parts:
        artifact = artifacts[part.data_id]
        raw_content = read_json_file(artifact.content_list_path)
        if isinstance(raw_content, dict) and isinstance(raw_content.get("content_list"), list):
            raw_content = raw_content["content_list"]
        content_list.extend(adjust_content_list_pages(raw_content, part.page_offset))
        middle = adjust_middle_json_pages(read_json_file(artifact.middle_json_path), part.page_offset)
        if isinstance(middle, dict) and isinstance(middle.get("pdf_info"), list):
            pdf_info.extend(middle["pdf_info"])
        source_parts.append(
            {
                "upload_name": part.upload_name,
                "page_start": part.page_start,
                "page_end": part.page_end,
                "page_offset": part.page_offset,
                "content_list_path": artifact.content_list_path,
                "middle_json_path": artifact.middle_json_path,
                "zip_path": artifact.zip_path,
            }
        )
    combined: dict[str, Any] = {"content_list": content_list, "source_parts": source_parts}
    if pdf_info:
        combined["pdf_info"] = pdf_info
    return combined


def build_merged_metadata(merged_markdown: str, success_parts: list[UploadPart], artifacts: dict[str, MineruPartArtifact]) -> dict[str, Any]:
    page_spans: list[dict[str, int]] = []
    semantic_spans: list[dict[str, Any]] = []
    search_cursor = 0
    for part in success_parts:
        artifact = artifacts[part.data_id]
        start_offset = merged_markdown.find(artifact.markdown, search_cursor)
        if start_offset < 0:
            start_offset = search_cursor
        search_cursor = start_offset + len(artifact.markdown)
        content_list = read_json_file(artifact.content_list_path)
        if isinstance(content_list, dict) and isinstance(content_list.get("content_list"), list):
            content_list = content_list["content_list"]
        middle_json = read_json_file(artifact.middle_json_path)
        metadata = build_mineru_metadata_from_json(
            artifact.markdown,
            {
                **(middle_json if isinstance(middle_json, dict) else {}),
                "content_list": content_list if isinstance(content_list, list) else [],
            },
        )
        for span in metadata.get("page_spans") or []:
            page = try_int(span.get("page"))
            start = try_int(span.get("start"))
            end = try_int(span.get("end"))
            if page is None or start is None or end is None:
                continue
            page_spans.append({"page": page + part.page_offset, "start": start + start_offset, "end": end + start_offset})
        for span in metadata.get("semantic_spans") or []:
            start = try_int(span.get("start"))
            end = try_int(span.get("end"))
            if start is None or end is None:
                continue
            copied = dict(span)
            copied["start"] = start + start_offset
            copied["end"] = end + start_offset
            page = try_int(copied.get("page"))
            if page is not None:
                copied["page"] = page + part.page_offset
            semantic_spans.append(copied)
    return {
        "page_spans": sorted(page_spans, key=lambda item: (item["page"], item["start"])),
        "semantic_spans": sorted(semantic_spans, key=lambda item: (item["start"], item["end"])),
        "source": "mineru",
        "format": "rageval_mineru_metadata_v1",
    }


def get_pdf_page_count(path: Path) -> int:
    import pypdf

    reader = pypdf.PdfReader(str(path))
    return len(reader.pages)


def split_pdf(path: Path, output_dir: Path, max_pages: int = MAX_PAGES_PER_TASK) -> list[tuple[Path, int, int]]:
    import pypdf

    reader = pypdf.PdfReader(str(path))
    page_count = len(reader.pages)
    if page_count <= max_pages:
        return [(path, 1, page_count)]

    parts: list[tuple[Path, int, int]] = []
    for start in range(0, page_count, max_pages):
        end = min(start + max_pages, page_count)
        writer = pypdf.PdfWriter()
        for page_index in range(start, end):
            writer.add_page(reader.pages[page_index])
        part_path = output_dir / f"{path.stem}_part{len(parts) + 1}_{start + 1}-{end}.pdf"
        with part_path.open("wb") as f:
            writer.write(f)
        parts.append((part_path, start + 1, end))
    return parts


def prepare_upload_parts(paths: list[Path], work_dir: Path, progress: ProgressCallback | None = None) -> list[UploadPart]:
    split_dir = work_dir / "mineru_parts"
    split_dir.mkdir(parents=True, exist_ok=True)
    upload_parts: list[UploadPart] = []

    for original_path in paths:
        ext = original_path.suffix.lower()
        if ext not in MINERU_SUPPORTED_EXTENSIONS:
            continue
        if not original_path.exists():
            raise FileNotFoundError(original_path)

        file_parts: list[tuple[Path, int, int]]
        if ext == ".pdf":
            page_count = get_pdf_page_count(original_path)
            if progress:
                progress(f"{original_path.name}: PDF page count {page_count}")
            file_parts = split_pdf(original_path, split_dir, MAX_PAGES_PER_TASK)
        else:
            file_parts = [(original_path, 1, 1)]

        total = len(file_parts)
        base_id = safe_data_id(original_path.name)
        for idx, (part_path, page_start, page_end) in enumerate(file_parts, 1):
            upload_parts.append(
                UploadPart(
                    original_path=str(original_path),
                    upload_path=str(part_path),
                    upload_name=part_path.name,
                    data_id=f"{base_id}_p{idx}",
                    part_index=idx,
                    part_total=total,
                    page_start=page_start,
                    page_end=page_end,
                    page_offset=page_start - 1,
                    size_bytes=part_path.stat().st_size,
                )
            )
    return upload_parts


def group_batches(parts: list[UploadPart]) -> list[list[UploadPart]]:
    batches: list[list[UploadPart]] = []
    current: list[UploadPart] = []
    current_bytes = 0
    for part in parts:
        would_exceed_count = len(current) >= MAX_FILES_PER_BATCH
        would_exceed_bytes = bool(current) and current_bytes + part.size_bytes > MAX_UPLOAD_BATCH_BYTES
        if would_exceed_count or would_exceed_bytes:
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(part)
        current_bytes += part.size_bytes
    if current:
        batches.append(current)
    return batches


def model_group_key(part: UploadPart) -> str:
    ext = Path(part.upload_name).suffix.lower()
    return "html" if ext in {".html", ".htm"} else "document"


def process_with_mineru(
    input_paths: list[Path],
    output_dir: Path,
    options: MineruOptions,
    progress: ProgressCallback | None = None,
) -> list[MineruParsedFile]:
    output_dir.mkdir(parents=True, exist_ok=True)
    client = MineruClient(options)
    parts = prepare_upload_parts(input_paths, output_dir, progress)
    if not parts:
        return []

    grouped: dict[str, list[UploadPart]] = {"document": [], "html": []}
    for part in parts:
        grouped[model_group_key(part)].append(part)

    completed_artifact_by_part: dict[str, MineruPartArtifact] = {}
    failed_by_part: dict[str, str] = {}
    part_by_data_id = {part.data_id: part for part in parts}

    for group_name, group_parts in grouped.items():
        if not group_parts:
            continue
        model_version = options.html_model_version if group_name == "html" else options.model_version
        for batch_index, batch in enumerate(group_batches(group_parts), 1):
            if progress:
                progress(f"MinerU: requesting upload URLs for {group_name} batch {batch_index}, files={len(batch)}")
            upload_response = client.create_batch_upload_urls(batch, model_version)
            batch_id = upload_response["data"]["batch_id"]
            upload_urls = upload_response["data"]["file_urls"]

            for part, upload_url in zip(batch, upload_urls):
                if progress:
                    progress(f"MinerU: uploading {part.upload_name}")
                client.upload_file(upload_url, part.upload_path)

            if progress:
                progress(f"MinerU: polling batch {batch_id}")
            poll_batch_until_done(client, batch_id, part_by_data_id, completed_artifact_by_part, failed_by_part, output_dir, progress)

    results: list[MineruParsedFile] = []
    parts_by_original: dict[str, list[UploadPart]] = {}
    for part in parts:
        parts_by_original.setdefault(part.original_path, []).append(part)

    for original_path, original_parts in parts_by_original.items():
        original_parts.sort(key=lambda p: p.part_index)
        success_parts = [p for p in original_parts if p.data_id in completed_artifact_by_part]
        failed_parts = [p for p in original_parts if p.data_id in failed_by_part]
        if not success_parts:
            results.append(
                MineruParsedFile(
                    original_path=original_path,
                    success=False,
                    part_count=len(original_parts),
                    error="; ".join(failed_by_part.get(p.data_id, "not completed") for p in original_parts),
                    parts=[asdict(p) for p in original_parts],
                )
            )
            continue

        merged_sections: list[str] = []
        for part in success_parts:
            title = f"<!-- source: {Path(original_path).name}; pages: {part.page_start}-{part.page_end}; part: {part.part_index}/{part.part_total} -->"
            merged_sections.append(f"{title}\n\n{completed_artifact_by_part[part.data_id].markdown}")
        merged = "\n\n".join(merged_sections).strip()
        out_name = f"{Path(original_path).stem}.mineru.md"
        markdown_path = unique_path(output_dir / out_name)
        markdown_path.write_text(merged, encoding="utf-8")
        metadata = build_merged_metadata(merged, success_parts, completed_artifact_by_part)
        metadata_path = markdown_path.with_suffix(markdown_path.suffix + ".meta.json")
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        combined_json = build_combined_mineru_json(success_parts, completed_artifact_by_part)
        combined_json_path = markdown_path.with_suffix(markdown_path.suffix + ".mineru.json")
        combined_json_path.write_text(json.dumps(combined_json, ensure_ascii=False), encoding="utf-8")
        zip_paths = [completed_artifact_by_part[p.data_id].zip_path for p in success_parts]
        json_paths = [path for p in success_parts for path in completed_artifact_by_part[p.data_id].json_paths]
        content_list_paths = [
            completed_artifact_by_part[p.data_id].content_list_path
            for p in success_parts
            if completed_artifact_by_part[p.data_id].content_list_path
        ]
        middle_json_paths = [
            completed_artifact_by_part[p.data_id].middle_json_path
            for p in success_parts
            if completed_artifact_by_part[p.data_id].middle_json_path
        ]
        error = "; ".join(failed_by_part[p.data_id] for p in failed_parts if p.data_id in failed_by_part) or None
        results.append(
            MineruParsedFile(
                original_path=original_path,
                success=True,
                markdown_path=str(markdown_path),
                metadata_path=str(metadata_path),
                zip_paths=zip_paths,
                json_paths=json_paths,
                content_list_paths=content_list_paths,
                middle_json_paths=middle_json_paths,
                combined_json_path=str(combined_json_path),
                text_chars=len(merged),
                part_count=len(original_parts),
                error=f"partial MinerU failure: {error}" if error else None,
                parts=[asdict(p) for p in original_parts],
            )
        )
    return results


def poll_batch_until_done(
    client: MineruClient,
    batch_id: str,
    part_by_data_id: dict[str, UploadPart],
    completed_artifact_by_part: dict[str, MineruPartArtifact],
    failed_by_part: dict[str, str],
    output_dir: Path,
    progress: ProgressCallback | None,
) -> None:
    deadline = time.time() + POLL_TIMEOUT_SECONDS
    seen_done: set[str] = set()
    while time.time() < deadline:
        status = client.get_batch_status(batch_id)
        extract_results = status.get("data", {}).get("extract_result") or []
        if not extract_results:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        for item in extract_results:
            data_id = item.get("data_id") or ""
            if not data_id:
                data_id = infer_data_id_by_filename(item.get("file_name"), part_by_data_id)
            if not data_id or data_id in seen_done:
                continue
            state = item.get("state")
            if state == "done":
                zip_url = item.get("full_zip_url")
                if not zip_url:
                    failed_by_part[data_id] = "done but missing full_zip_url"
                else:
                    part = part_by_data_id.get(data_id)
                    if progress and part:
                        progress(f"MinerU: downloading result {part.upload_name}")
                    if not part:
                        failed_by_part[data_id] = "done but upload part cannot be matched"
                    else:
                        completed_artifact_by_part[data_id] = client.download_artifacts(zip_url, part, output_dir)
                seen_done.add(data_id)
            elif state == "failed":
                failed_by_part[data_id] = item.get("err_msg") or "MinerU parse failed"
                seen_done.add(data_id)
            elif progress and item.get("extract_progress"):
                p = item["extract_progress"]
                progress(
                    f"MinerU: {item.get('file_name') or data_id} running "
                    f"{p.get('extracted_pages', 0)}/{p.get('total_pages', '?')} pages"
                )

        expected_ids = {
            data_id for data_id, part in part_by_data_id.items()
            if data_id in [r.get("data_id") for r in extract_results] or safe_mineru_filename(part.upload_name) in [r.get("file_name") for r in extract_results]
        }
        if expected_ids and all(data_id in completed_artifact_by_part or data_id in failed_by_part for data_id in expected_ids):
            return
        time.sleep(POLL_INTERVAL_SECONDS)

    raise MineruError(f"MinerU batch timeout: {batch_id}")


def infer_data_id_by_filename(file_name: str | None, part_by_data_id: dict[str, UploadPart]) -> str:
    if not file_name:
        return ""
    for data_id, part in part_by_data_id.items():
        if file_name in {part.upload_name, safe_mineru_filename(part.upload_name)}:
            return data_id
    return ""


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for idx in range(1, 10_000):
        candidate = path.with_name(f"{stem}_{idx}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"cannot allocate unique path for {path}")


def mineru_supported_paths(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.suffix.lower() in MINERU_SUPPORTED_EXTENSIONS]
