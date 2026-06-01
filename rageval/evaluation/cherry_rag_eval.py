"""
End-to-end QA evaluation against Cherry Studio's public local API.

This evaluator treats Cherry Studio as a black-box RAG target:
1. Search one or more pre-built Cherry knowledge bases.
2. Send the retrieved chunks plus the gold query to Cherry's chat API.
3. Score the final answer against the dataset's gold answers.

It intentionally does not upload documents. The expected workflow is that the
user has already prepared and vectorized the knowledge base in Cherry Studio.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from rageval.config import DATA_DIR, RESULT_ZH_DIR
from rageval.evaluation.deepseekmine_rag_eval import (
    answers_for,
    evidence_quotes_for,
    filter_rows_by_ids,
    hit_matches_evidence,
    is_answer_correct,
    load_jsonl,
    parse_ids,
    score_answer_with_fallback,
)


RESULT_DIR = RESULT_ZH_DIR


class CherryApiError(RuntimeError):
    pass


@dataclass
class CherryEvalOptions:
    dataset_name: str
    dataset_path: str = ""
    api_base: str = "http://127.0.0.1:23333"
    api_key: str = ""
    knowledge_base_ids: str = ""
    model: str = ""
    document_count: int = 5
    limit: int = 0
    ids: str = ""
    output: str = ""
    timeout: int = 300
    judge_mode: str = "rule_then_model"
    temperature: float = 0.0
    max_tokens: int = 1024


def read_cherry_key(raw_key: str = "") -> str:
    key = str(raw_key or "").strip() or os.environ.get("CHERRY_API_KEY", "").strip()
    if key:
        return key
    raise RuntimeError("Cherry API key is missing. Set CHERRY_API_KEY or pass --api-key.")


def cherry_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def request_json(
    method: str,
    api_base: str,
    path: str,
    api_key: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    url = api_base.rstrip("/") + path
    try:
        response = requests.request(
            method,
            url,
            headers=cherry_headers(api_key),
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise CherryApiError(f"Cherry API request failed: {method} {url}: {exc}") from exc
    body = response.text[:1000].replace("\n", " ")
    if response.status_code >= 400:
        raise CherryApiError(f"Cherry API returned HTTP {response.status_code}: {body}")
    try:
        data = response.json()
    except ValueError as exc:
        raise CherryApiError(f"Cherry API did not return JSON: {body}") from exc
    if not isinstance(data, dict):
        raise CherryApiError(f"Cherry API returned non-object JSON: {type(data).__name__}")
    if isinstance(data.get("error"), dict):
        error = data["error"]
        message = str(error.get("message") or error)
        raise CherryApiError(f"Cherry API error: {message}")
    return data


def resolve_dataset_path(dataset_name: str, dataset_path: str = "") -> Path:
    path = Path(dataset_path) if dataset_path else DATA_DIR / f"{dataset_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path


def split_ids(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,，\s]+", str(value or "")) if part.strip()]


def list_knowledge_bases(api_base: str, api_key: str, timeout: int) -> list[dict[str, Any]]:
    data = request_json("GET", api_base, "/v1/knowledge-bases?limit=100", api_key, timeout=timeout)
    bases = data.get("knowledge_bases") or data.get("data") or []
    return bases if isinstance(bases, list) else []


def list_models(api_base: str, api_key: str, timeout: int) -> list[dict[str, Any]]:
    data = request_json("GET", api_base, "/v1/models?limit=100", api_key, timeout=timeout)
    models = data.get("data") or []
    return models if isinstance(models, list) else []


def choose_default_model(models: list[dict[str, Any]]) -> str:
    blocked = re.compile(r"embedding|rerank|bge", re.I)
    preferred = re.compile(r"qwen3-8b|deepseek-v3|qwen2\.5-7b", re.I)
    candidates = [str(item.get("id") or "").strip() for item in models if str(item.get("id") or "").strip()]
    for model in candidates:
        if preferred.search(model) and not blocked.search(model):
            return model
    for model in candidates:
        if not blocked.search(model):
            return model
    raise RuntimeError("No usable chat model found from Cherry /v1/models.")


def search_knowledge(
    api_base: str,
    api_key: str,
    query: str,
    knowledge_base_ids: list[str],
    document_count: int,
    timeout: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "document_count": max(1, min(int(document_count), 20)),
    }
    if knowledge_base_ids:
        payload["knowledge_base_ids"] = knowledge_base_ids
    return request_json("POST", api_base, "/v1/knowledge-bases/search", api_key, payload=payload, timeout=timeout)


def build_answer_prompt(query: str, results: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for idx, result in enumerate(results, start=1):
        content = str(result.get("pageContent") or "").strip()
        if not content:
            continue
        source = result.get("metadata") or {}
        source_text = ""
        if isinstance(source, dict):
            source_text = str(source.get("source") or source.get("file") or source.get("path") or "").strip()
        kb_name = str(result.get("knowledge_base_name") or "").strip()
        title = " / ".join(part for part in [kb_name, source_text] if part)
        heading = f"[{idx}] {title}" if title else f"[{idx}]"
        chunks.append(f"{heading}\n{content}")
    context = "\n\n".join(chunks).strip() or "未检索到可用资料。"
    return f"""
请只根据下面的资料回答问题。要求：
1. 答案尽量简短，直接给出结论。
2. 如果资料不足以回答，请回答“资料不足”。
3. 不要使用资料以外的常识补全。

资料：
{context}

问题：
{query}
""".strip()


def chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是一个严谨的中文问答助手。只根据用户提供的资料回答。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        data = request_json("POST", api_base, "/v1/chat/completions", api_key, payload=payload, timeout=timeout)
        choices = data.get("choices") or []
        if not choices or not isinstance(choices, list):
            raise CherryApiError(f"Cherry chat response missing choices: {data}")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise CherryApiError(f"Cherry chat response missing message: {data}")
        return str(message.get("content") or message.get("reasoning_content") or "").strip()
    except CherryApiError as exc:
        if "does not support Chat Completions API" not in str(exc):
            raise
    return messages_completion(api_base, api_key, model, prompt, timeout, max_tokens, temperature)


def messages_completion(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
) -> str:
    payload = {
        "model": model,
        "system": "你是一个严谨的中文问答助手。只根据用户提供的资料回答。",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = request_json("POST", api_base, "/v1/messages", api_key, payload=payload, timeout=timeout)
    content = data.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        raise CherryApiError(f"Cherry messages response missing content: {data}")
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and str(item.get("text") or "").strip():
            parts.append(str(item["text"]).strip())
    if parts:
        return "\n".join(parts).strip()
    raise CherryApiError(f"Cherry messages response has no text content: {data}")


def score_retrieval(results: list[dict[str, Any]], gold_quotes: list[str], answers: list[str]) -> dict[str, Any]:
    hit_texts = [str(item.get("pageContent") or "") for item in results]
    matched_ranks: list[int] = []
    for quote in gold_quotes:
        for idx, hit_text in enumerate(hit_texts, start=1):
            if hit_matches_evidence(hit_text, quote, answers):
                matched_ranks.append(idx)
                break
    first_rank = min(matched_ranks) if matched_ranks else None
    return {
        "evidence_hit": bool(matched_ranks),
        "evidence_recall": min(len(matched_ranks) / max(1, len(gold_quotes)), 1.0),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "first_evidence_rank": first_rank,
        "retrieved_count": len(hit_texts),
    }


def score_answer(
    query: str,
    answers: list[str],
    prediction: str,
    gold_quotes: list[str],
    judge_mode: str,
    timeout: int,
) -> tuple[bool, str, dict[str, Any] | None]:
    if judge_mode == "rule":
        return is_answer_correct(prediction, answers), "rule", None
    return score_answer_with_fallback(query, answers, prediction, gold_quotes, timeout)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if not total:
        return {
            "total": 0,
            "qa_accuracy": 0.0,
            "qa_correct": 0,
            "qa_total": 0,
            "evidence_hit_rate": 0.0,
            "evidence_recall": 0.0,
            "mrr": 0.0,
            "avg_retrieved_count": 0.0,
        }
    qa_correct = sum(1 for item in results if item.get("answer_correct"))
    retrievals = [item.get("retrieval") or {} for item in results]
    return {
        "total": total,
        "qa_accuracy": qa_correct / total,
        "qa_correct": qa_correct,
        "qa_total": total,
        "evidence_hit_rate": sum(1 for r in retrievals if r.get("evidence_hit")) / total,
        "evidence_recall": sum(float(r.get("evidence_recall") or 0.0) for r in retrievals) / total,
        "mrr": sum(float(r.get("mrr") or 0.0) for r in retrievals) / total,
        "avg_retrieved_count": sum(int(r.get("retrieved_count") or 0) for r in retrievals) / total,
    }


def run_eval(options: CherryEvalOptions, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    log = progress or (lambda _msg: None)
    api_key = read_cherry_key(options.api_key)
    dataset_path = resolve_dataset_path(options.dataset_name, options.dataset_path)
    rows = filter_rows_by_ids(load_jsonl(dataset_path, options.limit), parse_ids(options.ids))
    if not rows:
        raise RuntimeError("Dataset has no rows")

    bases = list_knowledge_bases(options.api_base, api_key, min(options.timeout, 60))
    knowledge_base_ids = split_ids(options.knowledge_base_ids)
    if not knowledge_base_ids:
        if len(bases) == 1 and str(bases[0].get("id") or "").strip():
            knowledge_base_ids = [str(bases[0]["id"]).strip()]
        else:
            raise RuntimeError("Pass --knowledge-base-id when Cherry has zero or multiple knowledge bases.")

    models = list_models(options.api_base, api_key, min(options.timeout, 60))
    model = options.model.strip() or choose_default_model(models)
    knowledge_names = {
        str(base.get("id")): str(base.get("name") or "")
        for base in bases
        if str(base.get("id") or "").strip()
    }

    results: list[dict[str, Any]] = []
    stopped_reason = ""
    for idx, row in enumerate(rows, start=1):
        query = str(row.get("query", "")).strip()
        answers = answers_for(row)
        gold_quotes = evidence_quotes_for(row)
        log(f"[{idx}/{len(rows)}] Cherry 搜索：{query[:80]}")
        item: dict[str, Any] = {
            "id": row.get("id"),
            "query": query,
            "answer": answers,
            "gold_evidence": gold_quotes,
        }
        try:
            search_payload = search_knowledge(
                options.api_base,
                api_key,
                query,
                knowledge_base_ids,
                options.document_count,
                options.timeout,
            )
            search_results = search_payload.get("results") or []
            if not isinstance(search_results, list):
                search_results = []
            retrieval = score_retrieval(search_results, gold_quotes, answers)
            prompt = build_answer_prompt(query, search_results)
            prediction = chat_completion(
                options.api_base,
                api_key,
                model,
                prompt,
                options.timeout,
                options.max_tokens,
                options.temperature,
            )
            correct, method, judge = score_answer(
                query,
                answers,
                prediction,
                gold_quotes,
                options.judge_mode,
                options.timeout,
            )
            item.update(
                {
                    "retrieval": retrieval,
                    "prediction": prediction,
                    "answer_correct": correct,
                    "answer_judge_method": method,
                    "search_results": search_results,
                    "prompt_chars": len(prompt),
                }
            )
            if judge is not None:
                item["answer_judge"] = judge
            log(f"[{idx}/{len(rows)}] 判定：{'正确' if correct else '未命中'}；检索 {retrieval['retrieved_count']} 条")
        except Exception as exc:
            item.update(
                {
                    "retrieval": {
                        "evidence_hit": False,
                        "evidence_recall": 0.0,
                        "mrr": 0.0,
                        "retrieved_count": 0,
                    },
                    "prediction": "",
                    "answer_correct": False,
                    "answer_judge_method": "error",
                    "answer_judge": {"correct": False, "reason": str(exc)},
                }
            )
            log(f"[{idx}/{len(rows)}] 错误：{str(exc)[:160]}")
        results.append(item)

    report = {
        "schema": "cherry-rag-e2e-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset_name": options.dataset_name,
        "dataset_path": str(dataset_path),
        "api_base": options.api_base,
        "knowledge_base_ids": knowledge_base_ids,
        "knowledge_bases": [{"id": kb_id, "name": knowledge_names.get(kb_id, "")} for kb_id in knowledge_base_ids],
        "model": model,
        "document_count": options.document_count,
        "judge_mode": options.judge_mode,
        "summary": summarize(results),
        "stopped_reason": stopped_reason,
        "completed": not stopped_reason and len(results) == len(rows),
        "results": results,
    }

    output = Path(options.output) if options.output else RESULT_DIR / f"cherry_qa_{options.dataset_name}_{int(time.time())}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report["output_path"] = str(output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Cherry Studio RAG QA accuracy with pre-built knowledge bases.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--api-base", default="http://127.0.0.1:23333")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--knowledge-base-id", dest="knowledge_base_ids", default="", help="One or more KB ids, comma-separated.")
    parser.add_argument("--model", default="", help="Cherry model id, e.g. silicon:Qwen/Qwen3-8B.")
    parser.add_argument("--document-count", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="", help="Comma-separated QA ids to evaluate.")
    parser.add_argument("--output", default="")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--judge-mode", choices=["rule", "rule_then_model"], default="rule_then_model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_eval(
        CherryEvalOptions(
            dataset_name=args.dataset,
            dataset_path=args.dataset_path,
            api_base=args.api_base.rstrip("/"),
            api_key=args.api_key,
            knowledge_base_ids=args.knowledge_base_ids,
            model=args.model,
            document_count=args.document_count,
            limit=args.limit,
            ids=args.ids,
            output=args.output,
            timeout=args.timeout,
            judge_mode=args.judge_mode,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        progress=lambda msg: print(f"[cherry] {msg}", flush=True),
    )
    summary = report["summary"]
    print("[cherry] 完成")
    print(f"[cherry] qa_accuracy={summary['qa_accuracy']:.4f} ({summary['qa_correct']}/{summary['qa_total']})")
    print(f"[cherry] evidence_hit_rate={summary['evidence_hit_rate']:.4f}")
    print(f"[cherry] mrr={summary['mrr']:.4f}")
    print(f"[cherry] report={report['output_path']}")


if __name__ == "__main__":
    main()
