"""
End-to-end RAG evaluation against a running deepseekmine service.

This is the business-flow evaluator: it uploads the generated corpus to
deepseekmine, calls /api/search for each gold question, sends the returned
result_prompt to MiMo, then scores both evidence retrieval and final answer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from rageval.config import DATA_DIR, MIMO_KEY_FILE, RESULT_ZH_DIR, TEMP_DOCS_DIR
from rageval.deepseekmine.compat_import import upload_markdown_assets_via_native_upload
from rageval.llm import (
    LLMProviderConfig,
    LLMQuotaExceededError,
    chat_completion,
    load_llm_config,
    message_text,
)


RESULT_DIR = RESULT_ZH_DIR
TEMP_DIR = TEMP_DOCS_DIR


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


class MimoQuotaExceededError(LLMQuotaExceededError):
    pass


def is_quota_error(status_code: int, body: str) -> bool:
    text = str(body or "").lower()
    if status_code == 402:
        return True
    return any(pattern in text for pattern in QUOTA_ERROR_PATTERNS)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", "", text).lower()


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    return text.strip()


def load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def parse_ids(ids_text: str) -> list[int]:
    ids: list[int] = []
    for part in str(ids_text or "").split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    return ids


def filter_rows_by_ids(rows: list[dict[str, Any]], ids: list[int]) -> list[dict[str, Any]]:
    if not ids:
        return rows
    wanted = set(ids)
    found = {int(row.get("id", -1)) for row in rows if str(row.get("id", "")).strip()}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"Unknown QA ids: {missing}")
    return [row for row in rows if int(row.get("id", -1)) in wanted]


def flatten(items: Any) -> list[Any]:
    if isinstance(items, list):
        out: list[Any] = []
        for item in items:
            out.extend(flatten(item))
        return out
    return [items]


def answers_for(row: dict[str, Any]) -> list[str]:
    return [str(x).strip() for x in flatten(row.get("answer", [])) if str(x).strip()]


def evidence_quotes_for(row: dict[str, Any]) -> list[str]:
    quotes: list[str] = []
    evidence = row.get("evidence")
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict) and str(item.get("quote", "")).strip():
                quotes.append(str(item["quote"]).strip())
            elif isinstance(item, str) and item.strip():
                quotes.append(item.strip())
    if not quotes:
        positives = row.get("positive", [])
        quotes.extend(str(x).strip() for x in flatten(positives) if str(x).strip())
    return quotes


def evidence_source_files_for(row: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    evidence = row.get("evidence")
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source_file", "")).strip()
            if not source:
                continue
            path = Path(source)
            if path.exists():
                paths.append(path)
    return paths


def build_fallback_corpus(rows: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for quote in evidence_quotes_for(row):
            key = normalize_text(quote)
            if key and key not in seen:
                seen.add(key)
                parts.append(quote)
    return "\n\n".join(parts)


def build_selected_corpus(rows: list[dict[str, Any]], dataset_name: str) -> Path:
    TEMP_DIR.mkdir(exist_ok=True)
    out_path = TEMP_DIR / f"deepseekmine_selected_{dataset_name}_{int(time.time())}.md"
    parts: list[str] = []
    seen_paths: set[str] = set()

    for row in rows:
        for path in evidence_source_files_for(row):
            key = str(path.resolve())
            if key in seen_paths:
                continue
            seen_paths.add(key)
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                text = ""
            if text:
                parts.append(f"# Source: {path.name}\n\n{text}")

    if not parts:
        parts.append(build_fallback_corpus(rows))

    out_path.write_text("\n\n---\n\n".join(part for part in parts if part.strip()), encoding="utf-8")
    return out_path


def resolve_dataset_paths(dataset_name: str, dataset_path: str = "", corpus_path: str = "") -> tuple[Path, Path | None]:
    dataset = Path(dataset_path) if dataset_path else DATA_DIR / f"{dataset_name}.json"
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    if corpus_path:
        corpus = Path(corpus_path)
        if not corpus.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus}")
        return dataset, corpus

    report_path = DATA_DIR / f"{dataset.stem}.json.report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for key in ("corpus_path", "corpus", "corpus_source"):
                value = str(report.get(key) or "").strip()
                if value:
                    corpus = Path(value)
                    if corpus.exists():
                        return dataset, corpus
        except Exception:
            pass

    for suffix in (".corpus.txt", ".corpus.md"):
        corpus = DATA_DIR / f"{dataset.stem}{suffix}"
        if corpus.exists():
            return dataset, corpus
    return dataset, None


def read_mimo_key() -> str:
    env_key = os.environ.get("MIMO_API_KEY", "").strip()
    if env_key:
        return env_key
    if MIMO_KEY_FILE.exists():
        key = MIMO_KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            os.environ["MIMO_API_KEY"] = key
            return key
    raise RuntimeError("MiMo API key is missing. Set MIMO_API_KEY or create E:\\RAG\\.mimo_api_key")


def load_answer_model_config() -> LLMProviderConfig:
    config = load_llm_config()
    if config.provider_id == "ollama":
        return config
    if config.provider_id == "mimo" and not config.api_key:
        config.api_key = read_mimo_key()
    if not config.api_key:
        raise RuntimeError(f"{config.provider_name} API key is missing. Configure it in the API settings page or set the provider environment variable.")
    return config


def mimo_chat_completion(messages: list[dict[str, str]], max_completion_tokens: int, timeout: int, purpose: str) -> dict[str, Any]:
    try:
        return chat_completion(messages, max_completion_tokens, timeout, purpose, load_answer_model_config())
    except LLMQuotaExceededError as exc:
        raise MimoQuotaExceededError(str(exc)) from exc


def health_check(api_base: str, timeout: int = 10) -> None:
    url = api_base.rstrip("/") + "/api/health"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code < 500:
            return
    except requests.RequestException:
        pass

    # Some dev builds may not expose /api/health. Verify with OPTIONS /api/search.
    try:
        resp = requests.options(api_base.rstrip("/") + "/api/search", timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"无法连接 deepseekmine 服务：{api_base}。请先启动 E:\\deepseekmine 项目，确认 {api_base} 能打开后再运行真实测评。"
        ) from exc
    if resp.status_code >= 500:
        raise RuntimeError(f"deepseekmine 服务异常：{api_base} 返回 HTTP {resp.status_code}，请确认项目已正常启动。")


def upload_corpus(api_base: str, knowledge_label: str, corpus_path: Path, timeout: int = 900) -> tuple[list[str], dict[str, Any]]:
    content_type = "text/plain" if corpus_path.suffix.lower() == ".txt" else "text/markdown"
    with corpus_path.open("rb") as handle:
        resp = requests.post(
            api_base.rstrip("/") + "/api/files/upload",
            data={
                "knowledgeLabel": knowledge_label,
                "waitForProcessing": "true",
                "fastParsing": "true",
            },
            files={"file": (corpus_path.name, handle, content_type)},
            timeout=timeout,
        )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"deepseekmine upload failed: {payload}")
    doc_ids = [str(x).strip() for x in payload.get("files", []) if str(x).strip()]
    if not doc_ids:
        doc_ids = [corpus_path.name]
    return doc_ids, payload


def load_asset_manifest(dataset_name: str) -> dict[str, Any] | None:
    manifest_path = DATA_DIR / "assets" / dataset_name / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def asset_markdown_paths(dataset_name: str) -> list[Path]:
    manifest = load_asset_manifest(dataset_name)
    if not manifest:
        return []
    paths: list[Path] = []
    for raw in manifest.get("mineru_markdown_files") or []:
        path = Path(str(raw))
        sidecar = path.with_suffix(path.suffix + ".mineru.json")
        if path.exists() and sidecar.exists():
            paths.append(path)
    return paths


def search_deepseekmine(
    api_base: str,
    query: str,
    knowledge_label: str,
    doc_ids: list[str],
    use_adaptive_rag: bool,
    timeout: int = 180,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "knowledgeLabel": knowledge_label,
        "reset": True,
        "useAdaptiveRag": use_adaptive_rag,
        "docIds": doc_ids,
    }
    resp = requests.post(api_base.rstrip("/") + "/api/search", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok", True) and data.get("error"):
        raise RuntimeError(f"deepseekmine search failed: {data}")
    return data


def answer_with_mimo(prompt: str, timeout: int = 300) -> str:
    data = mimo_chat_completion(
        [
            {
                "role": "system",
                "content": "你是一个严谨的中文问答助手。只根据用户提供的上下文回答，答案尽量简短，直接给出结论。",
            },
            {"role": "user", "content": prompt},
        ],
        2048,
        timeout,
        "answer",
    )
    return strip_thinking(message_text(data))


def is_answer_correct(prediction: str, answers: list[str]) -> bool:
    pred = normalize_answer_text(prediction)
    return bool(answers) and all(answer_item_correct(pred, answer) for answer in answers)


def normalize_answer_text(value: str) -> str:
    value = re.sub(r"【\d+】|\[\d+\]|\(\d+\)", "", str(value or ""))
    value = re.sub(r"(\d+)\.0+(?=%|[^\d]|$)", r"\1", value)
    value = normalize_text(value)
    return value


def compact_answer_text(value: str) -> str:
    return re.sub(r"[\s，,。；;：:、！!？?（）()\[\]【】《》“”\"'`·\-—_/\\]+", "", normalize_answer_text(value))


def answer_item_correct(prediction_norm: str, answer: str) -> bool:
    answer_norm = normalize_answer_text(answer)
    pred_compact = compact_answer_text(prediction_norm)
    answer_compact = compact_answer_text(answer_norm)
    if not answer_compact:
        return False
    if answer_compact in pred_compact:
        return True
    if len(answer_compact) <= 6:
        return False
    answer_chars = [char for char in answer_compact if re.search(r"[\u4e00-\u9fffA-Za-z0-9]", char)]
    if not answer_chars:
        return False
    matched_chars = sum(1 for char in answer_chars if char in pred_compact)
    char_recall = matched_chars / len(answer_chars)
    return char_recall >= 0.72


def _strip_json_fences(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _repair_json_candidate(text: str) -> str:
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return re.sub(r'\\(?!["\\/bfnrtu])', lambda _: r"\\", text)


def _balanced_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    start = -1
    depth = 0
    in_string = False
    escape = False
    for idx, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start : idx + 1])
                start = -1
    return candidates


def extract_json_object(text: str) -> dict[str, Any]:
    text = _strip_json_fences(text)
    candidates = [text]
    candidates.extend(candidate for candidate in _balanced_json_candidates(text) if candidate not in candidates)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        if snippet not in candidates:
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
    raise ValueError(f"judge did not return a JSON object: {last_error}")


def _json_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "correct", "对", "正确"}
    return False


def extract_judge_payload_or_retry(raw_text: str, timeout: int) -> dict[str, Any]:
    try:
        return extract_json_object(raw_text)
    except ValueError:
        repair_prompt = f"""
Convert the following judge output into one valid JSON object.
Return only JSON. No markdown. No explanation.
Schema: {{"correct": true, "reason": "short reason"}}

Original output:
{raw_text[:4000]}
""".strip()
        repair_data = mimo_chat_completion(
            [
                {"role": "system", "content": "You repair malformed JSON. Return only one valid JSON object."},
                {"role": "user", "content": repair_prompt},
            ],
            256,
            timeout,
            "judge json repair",
        )
        return extract_json_object(strip_thinking(message_text(repair_data)))


def judge_answer_with_mimo(
    query: str,
    answers: list[str],
    prediction: str,
    gold_quotes: list[str],
    timeout: int = 300,
) -> dict[str, Any]:
    prompt = f"""
你是 RAG 测评裁判。请判断“模型回答”是否覆盖“标准答案”的全部关键要点。

判定规则：
1. 不要求逐字一致，允许同义改写、顺序变化、引用编号和少量无害补充。
2. 数字、日期、人名、药名、方名、列表项等关键事实不能错漏。
3. 标准答案有多个并列要点时，模型回答必须覆盖全部核心要点。
4. 如果模型回答与标准答案矛盾、缺少关键项、答非所问，判为 false。
5. 只输出 JSON，不要输出额外文字。

输出格式：
{{"correct": true, "reason": "简短理由"}}

问题：
{query}

标准答案：
{json.dumps(answers, ensure_ascii=False)}

标准证据：
{json.dumps(gold_quotes[:3], ensure_ascii=False)}

模型回答：
{prediction}
""".strip()
    data = mimo_chat_completion(
        [
            {
                "role": "system",
                "content": "你是严格、稳定的 RAG 答案测评裁判，只输出 JSON。",
            },
            {"role": "user", "content": prompt},
        ],
        512,
        timeout,
        "judge",
    )
    payload = extract_judge_payload_or_retry(strip_thinking(message_text(data)), timeout)
    return {
        "correct": _json_bool(payload.get("correct")),
        "reason": str(payload.get("reason") or "").strip(),
    }


def score_answer_with_fallback(
    query: str,
    answers: list[str],
    prediction: str,
    gold_quotes: list[str],
    timeout: int = 300,
) -> tuple[bool, str, dict[str, Any] | None]:
    if is_answer_correct(prediction, answers):
        return True, "rule", None
    judge = judge_answer_with_mimo(query, answers, prediction, gold_quotes, timeout)
    return bool(judge.get("correct")), "model_judge", judge


def hit_matches_evidence(hit_text: str, quote: str, answers: list[str]) -> bool:
    hit_norm = normalize_text(hit_text)
    quote_norm = normalize_text(quote)
    if not hit_norm or not quote_norm:
        return False
    if quote_norm in hit_norm or hit_norm in quote_norm:
        return True
    if len(quote_norm) >= 24 and quote_norm[:24] in hit_norm:
        return True
    return bool(answers) and all(normalize_text(answer) in hit_norm for answer in answers)


def score_retrieval(hits: list[dict[str, Any]], gold_quotes: list[str], answers: list[str]) -> dict[str, Any]:
    hit_texts = [str(hit.get("content", "") or "") for hit in hits]
    matched_ranks: list[int] = []
    for quote in gold_quotes:
        for idx, hit_text in enumerate(hit_texts, start=1):
            if hit_matches_evidence(hit_text, quote, answers):
                matched_ranks.append(idx)
                break

    matched_count = len(matched_ranks)
    gold_count = max(1, len(gold_quotes))
    hit_count = len(hit_texts)
    first_rank = min(matched_ranks) if matched_ranks else None
    return {
        "evidence_hit": matched_count > 0,
        "evidence_recall": min(matched_count / gold_count, 1.0),
        "evidence_precision": matched_count / hit_count if hit_count else 0.0,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "first_evidence_rank": first_rank,
        "retrieved_count": hit_count,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if not total:
        return {
            "total": 0,
            "evidence_hit_rate": 0.0,
            "evidence_recall": 0.0,
            "mrr": 0.0,
            "qa_accuracy": 0.0,
            "qa_correct": 0,
            "qa_answer_errors": 0,
            "qa_quota_errors": 0,
            "avg_retrieved_count": 0.0,
        }
    qa_results = [r for r in results if "answer_correct" in r]
    qa_correct = sum(1 for r in qa_results if r.get("answer_correct"))
    qa_answer_errors = sum(1 for r in qa_results if r.get("answer_judge_method") == "mimo_answer_error")
    qa_quota_errors = sum(1 for r in qa_results if r.get("answer_judge_method") == "api_quota_exhausted")
    return {
        "total": total,
        "evidence_hit_rate": sum(1 for r in results if r["retrieval"]["evidence_hit"]) / total,
        "evidence_recall": sum(float(r["retrieval"]["evidence_recall"]) for r in results) / total,
        "mrr": sum(float(r["retrieval"]["mrr"]) for r in results) / total,
        "qa_accuracy": qa_correct / len(qa_results) if qa_results else 0.0,
        "qa_correct": qa_correct,
        "qa_total": len(qa_results),
        "qa_answer_errors": qa_answer_errors,
        "qa_quota_errors": qa_quota_errors,
        "avg_retrieved_count": sum(int(r["retrieval"]["retrieved_count"]) for r in results) / total,
    }


@dataclass
class EvalOptions:
    dataset_name: str
    dataset_path: str = ""
    corpus_path: str = ""
    api_base: str = "http://127.0.0.1:3335"
    knowledge_label: str = ""
    mode: str = "qa"
    limit: int = 0
    ids: str = ""
    output: str = ""
    include_prompts: bool = True
    use_adaptive_rag: bool = True
    timeout: int = 900


def run_eval(options: EvalOptions, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    log = progress or (lambda msg: None)
    dataset_path, corpus_path = resolve_dataset_paths(options.dataset_name, options.dataset_path, options.corpus_path)
    rows = filter_rows_by_ids(load_jsonl(dataset_path, options.limit), parse_ids(options.ids))
    if not rows:
        raise RuntimeError("Dataset has no rows")

    if not options.knowledge_label:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        options.knowledge_label = f"rag_eval_{dataset_path.stem}_{stamp}"

    selected_ids = parse_ids(options.ids)
    selected_corpus_path: Path | None = None
    if selected_ids:
        selected_corpus_path = build_selected_corpus(rows, dataset_path.stem)
        corpus_path = selected_corpus_path
        log(f"已按所选 QA 构建独立测试语料：{corpus_path}")
    elif corpus_path is None:
        fallback_path = DATA_DIR / f"{dataset_path.stem}.corpus.generated.md"
        fallback_path.write_text(build_fallback_corpus(rows), encoding="utf-8")
        corpus_path = fallback_path
        log(f"未找到 corpus 文件，使用 evidence 生成临时语料：{corpus_path}")

    log(f"检查 deepseekmine 服务：{options.api_base}")
    health_check(options.api_base)

    local_asset_paths = [] if selected_ids else asset_markdown_paths(dataset_path.stem)
    if local_asset_paths:
        requested_knowledge_label = options.knowledge_label
        log(f"创建 deepseekmine 原生知识库，并上传 MinerU Markdown + sidecar：{len(local_asset_paths)} 个文件")
        imported_knowledge_label, doc_ids, upload_payload = upload_markdown_assets_via_native_upload(
            options.api_base,
            requested_knowledge_label,
            local_asset_paths,
            options.timeout,
            progress=log,
        )
        options.knowledge_label = imported_knowledge_label
        log(f"deepseekmine 原生知识库创建完成：请求标签 {requested_knowledge_label} -> 实际 knowledgeLabel {options.knowledge_label}")
    else:
        log(f"上传评测语料到 deepseekmine：{corpus_path.name}")
        doc_ids, upload_payload = upload_corpus(options.api_base, options.knowledge_label, corpus_path, options.timeout)
    log(f"上传完成，docIds：{', '.join(doc_ids)}")

    if options.mode == "qa":
        load_answer_model_config()
        answer_config = load_llm_config()

    results: list[dict[str, Any]] = []
    stopped_reason = ""
    for idx, row in enumerate(rows, start=1):
        query = str(row.get("query", "")).strip()
        answers = answers_for(row)
        gold_quotes = evidence_quotes_for(row)
        log(f"[{idx}/{len(rows)}] 检索：{query[:80]}")
        search_result = search_deepseekmine(
            options.api_base,
            query,
            options.knowledge_label,
            doc_ids,
            options.use_adaptive_rag,
            options.timeout,
        )
        hits = search_result.get("hits", [])
        if not isinstance(hits, list):
            hits = []
        retrieval = score_retrieval(hits, gold_quotes, answers)
        item: dict[str, Any] = {
            "id": row.get("id"),
            "query": query,
            "answer": answers,
            "gold_evidence": gold_quotes,
            "retrieval": retrieval,
            "hit_titles": [str(hit.get("title", "")) for hit in hits[:10] if isinstance(hit, dict)],
            "prompt_chars": len(str(search_result.get("result_prompt", ""))),
        }
        if options.include_prompts:
            item["result_prompt"] = search_result.get("result_prompt", "")
        if options.mode == "qa":
            try:
                prediction = answer_with_mimo(str(search_result.get("result_prompt", "")), options.timeout)
            except MimoQuotaExceededError as exc:
                stopped_reason = str(exc)
                prediction = ""
                item["prediction"] = prediction
                item["answer_correct"] = False
                item["answer_judge_method"] = "api_quota_exhausted"
                item["answer_judge"] = {"correct": False, "reason": stopped_reason}
                log(f"[{idx}/{len(rows)}] 回答模型额度不足或请求超限，已保护性中止：{stopped_reason[:160]}")
                results.append(item)
                break
            except Exception as exc:
                prediction = ""
                item["prediction"] = prediction
                item["answer_correct"] = False
                item["answer_judge_method"] = "mimo_answer_error"
                item["answer_judge"] = {"correct": False, "reason": str(exc)}
                log(f"[{idx}/{len(rows)}] 回答模型失败，已记为未命中并继续：{str(exc)[:160]}")
                results.append(item)
                continue
            item["prediction"] = prediction
            try:
                correct, method, judge = score_answer_with_fallback(
                    query,
                    answers,
                    prediction,
                    gold_quotes,
                    options.timeout,
                )
            except MimoQuotaExceededError as exc:
                stopped_reason = str(exc)
                item["answer_correct"] = False
                item["answer_judge_method"] = "api_quota_exhausted"
                item["answer_judge"] = {"correct": False, "reason": stopped_reason}
                log(f"[{idx}/{len(rows)}] 复核模型额度不足或请求超限，已保护性中止：{stopped_reason[:160]}")
                results.append(item)
                break
            except Exception as exc:
                correct, method, judge = False, "model_judge_error", {"correct": False, "reason": str(exc)}
            item["answer_correct"] = correct
            item["answer_judge_method"] = method
            if judge is not None:
                item["answer_judge"] = judge
            if method == "rule":
                log(f"[{idx}/{len(rows)}] 答案判定：正确")
            elif method == "model_judge":
                reason = str((judge or {}).get("reason") or "")
                suffix = f" - {reason[:80]}" if reason else ""
                log(f"[{idx}/{len(rows)}] 答案复核：{'正确' if correct else '未命中'}{suffix}")
            else:
                log(f"[{idx}/{len(rows)}] 答案复核失败：未命中 - {(judge or {}).get('reason', '')}")
        results.append(item)

    report = {
        "schema": "deepseekmine-rag-e2e-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset_name": options.dataset_name,
        "dataset_path": str(dataset_path),
        "corpus_path": str(corpus_path),
        "uploaded_asset_markdown_paths": [str(path) for path in local_asset_paths],
        "api_base": options.api_base,
        "knowledge_label": options.knowledge_label,
        "doc_ids": doc_ids,
        "mode": options.mode,
        "selected_ids": selected_ids,
        "answer_backend": answer_config.provider_id if options.mode == "qa" else None,
        "answer_model": answer_config.model if options.mode == "qa" else None,
        "answer_judge": "rule_then_model_fallback" if options.mode == "qa" else None,
        "upload": upload_payload,
        "summary": summarize(results),
        "stopped_reason": stopped_reason,
        "completed": not stopped_reason and len(results) == len(rows),
        "results": results,
    }
    if selected_corpus_path and selected_corpus_path.exists():
        report["selected_corpus_path"] = str(selected_corpus_path)

    output = Path(options.output) if options.output else RESULT_DIR / f"deepseekmine_{options.mode}_{options.dataset_name}_{int(time.time())}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report["output_path"] = str(output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--corpus-path", default="")
    parser.add_argument("--api-base", default="http://127.0.0.1:3335")
    parser.add_argument("--knowledge-label", default="")
    parser.add_argument("--mode", choices=["retrieval", "qa"], default="qa")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="", help="Comma-separated QA ids to evaluate")
    parser.add_argument("--output", default="")
    parser.add_argument("--no-prompts", action="store_true")
    parser.add_argument("--no-adaptive-rag", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    report = run_eval(
        EvalOptions(
            dataset_name=args.dataset,
            dataset_path=args.dataset_path,
            corpus_path=args.corpus_path,
            api_base=args.api_base.rstrip("/"),
            knowledge_label=args.knowledge_label,
            mode=args.mode,
            limit=args.limit,
            ids=args.ids,
            output=args.output,
            include_prompts=not args.no_prompts,
            use_adaptive_rag=not args.no_adaptive_rag,
            timeout=args.timeout,
        ),
        progress=lambda msg: print(f"[eval] {msg}", flush=True),
    )
    summary = report["summary"]
    print("[eval] 完成")
    print(f"[eval] evidence_hit_rate={summary['evidence_hit_rate']:.4f}")
    print(f"[eval] evidence_recall={summary['evidence_recall']:.4f}")
    print(f"[eval] mrr={summary['mrr']:.4f}")
    if report["mode"] == "qa":
        print(f"[eval] qa_accuracy={summary['qa_accuracy']:.4f} ({summary['qa_correct']}/{summary['qa_total']})")
    print(f"[eval] report={report['output_path']}")


if __name__ == "__main__":
    main()
