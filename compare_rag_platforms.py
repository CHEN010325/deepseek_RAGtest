from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rageval.config import RESULT_ZH_DIR
from rageval.evaluation.cherry_rag_eval import CherryEvalOptions, run_eval as run_cherry_eval
from rageval.evaluation import deepseekmine_rag_eval as deep_eval
from rageval.evaluation.deepseekmine_rag_eval import EvalOptions, run_eval as run_deepseekmine_eval
from rageval.llm import default_provider_config, load_llm_config


def pct(value: Any) -> float:
    try:
        return round(float(value) * 100, 2)
    except (TypeError, ValueError):
        return 0.0


def row_for(platform: str, report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") or {}
    return {
        "platform": platform,
        "dataset": report.get("dataset_name") or report.get("dataset") or "",
        "qa_accuracy_percent": pct(summary.get("qa_accuracy")),
        "qa_correct": summary.get("qa_correct", 0),
        "qa_total": summary.get("qa_total", summary.get("total", 0)),
        "evidence_hit_rate_percent": pct(summary.get("evidence_hit_rate")),
        "evidence_recall_percent": pct(summary.get("evidence_recall")),
        "mrr": round(float(summary.get("mrr") or 0.0), 4),
        "avg_retrieved_count": round(float(summary.get("avg_retrieved_count") or 0.0), 2),
        "answer_model": report.get("answer_model") or report.get("model") or "",
        "report_path": report.get("output_path") or "",
        "completed": report.get("completed", False),
        "stopped_reason": report.get("stopped_reason") or "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def install_deeplocal_answer_model_override(args: argparse.Namespace):
    saved_config = load_llm_config()
    if saved_config.provider_id == args.deeplocal_answer_provider:
        config = saved_config
    else:
        config = default_provider_config(args.deeplocal_answer_provider)
    if args.deeplocal_answer_api_url:
        config.api_url = args.deeplocal_answer_api_url
    if args.deeplocal_answer_model:
        config.model = args.deeplocal_answer_model
        if config.model not in config.models:
            config.models.insert(0, config.model)
    if args.deeplocal_answer_api_key:
        config.api_key = args.deeplocal_answer_api_key
    elif config.provider_id == "siliconflow" and os.environ.get("SILICONFLOW_API_KEY"):
        config.api_key = os.environ["SILICONFLOW_API_KEY"].strip()
    elif config.provider_id == "ollama":
        config.api_key = ""

    if config.provider_id != "ollama" and not config.api_key:
        raise RuntimeError(
            f"{config.provider_name} API key is missing for DeepLocal answer model. "
            f"Set the provider environment key or pass --deeplocal-answer-api-key."
        )

    original = deep_eval.load_llm_config
    deep_eval.load_llm_config = lambda: config
    return original


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DeepLocal/deepseekmine and Cherry Studio on the same RAG QA dataset.")
    parser.add_argument("--dataset", default="zh_int_clean")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--document-count", type=int, default=20)
    parser.add_argument("--deepseekmine-api-base", default="http://127.0.0.1:3335")
    parser.add_argument("--cherry-api-base", default="http://127.0.0.1:23333")
    parser.add_argument("--cherry-api-key", default="")
    parser.add_argument("--cherry-knowledge-base-id", default="")
    parser.add_argument("--cherry-model", default="silicon:deepseek-ai/DeepSeek-V4-Flash")
    parser.add_argument("--deeplocal-answer-provider", default="siliconflow")
    parser.add_argument("--deeplocal-answer-model", default="deepseek-ai/DeepSeek-V4-Flash")
    parser.add_argument("--deeplocal-answer-api-url", default="")
    parser.add_argument("--deeplocal-answer-api-key", default="")
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--no-prompts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cherry_api_key = args.cherry_api_key or os.environ.get("CHERRY_API_KEY", "")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = args.output_prefix or f"platform_compare_{args.dataset}_{stamp}"
    RESULT_ZH_DIR.mkdir(parents=True, exist_ok=True)
    deep_report_path = RESULT_ZH_DIR / f"{prefix}_deeplocal.json"
    cherry_report_path = RESULT_ZH_DIR / f"{prefix}_cherry.json"

    print(f"[compare] dataset={args.dataset} limit={args.limit or 'all'} ids={args.ids or '-'}", flush=True)
    print("[compare] running DeepLocal/deepseekmine and Cherry Studio in parallel...", flush=True)
    original_load_llm_config = install_deeplocal_answer_model_override(args)
    try:
        tasks = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            tasks[
                executor.submit(
                    run_deepseekmine_eval,
                    EvalOptions(
                        dataset_name=args.dataset,
                        api_base=args.deepseekmine_api_base.rstrip("/"),
                        mode="qa",
                        limit=args.limit,
                        ids=args.ids,
                        output=str(deep_report_path),
                        include_prompts=not args.no_prompts,
                        timeout=args.timeout,
                    ),
                    progress=lambda msg: print(f"[deeplocal] {msg}", flush=True),
                )
            ] = "DeepLocal"
            tasks[
                executor.submit(
                    run_cherry_eval,
                    CherryEvalOptions(
                        dataset_name=args.dataset,
                        api_base=args.cherry_api_base.rstrip("/"),
                        api_key=cherry_api_key,
                        knowledge_base_ids=args.cherry_knowledge_base_id,
                        model=args.cherry_model,
                        document_count=args.document_count,
                        limit=args.limit,
                        ids=args.ids,
                        output=str(cherry_report_path),
                        timeout=args.timeout,
                        judge_mode="rule_then_model",
                    ),
                    progress=lambda msg: print(f"[cherry] {msg}", flush=True),
                )
            ] = "Cherry Studio"

            reports: dict[str, dict[str, Any]] = {}
            for future in as_completed(tasks):
                platform = tasks[future]
                reports[platform] = future.result()
                print(f"[compare] {platform} finished", flush=True)

        deep_report = reports["DeepLocal"]
        cherry_report = reports["Cherry Studio"]
    finally:
        deep_eval.load_llm_config = original_load_llm_config

    rows = [
        row_for("DeepLocal", deep_report),
        row_for("Cherry Studio", cherry_report),
    ]
    summary_path = RESULT_ZH_DIR / f"{prefix}_summary.json"
    csv_path = RESULT_ZH_DIR / f"{prefix}_summary.csv"
    winner = max(rows, key=lambda item: item["qa_accuracy_percent"])
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": args.dataset,
        "limit": args.limit,
        "ids": args.ids,
        "winner_by_qa_accuracy": winner["platform"],
        "rows": rows,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)

    print("[compare] complete", flush=True)
    for row in rows:
        print(
            "[compare] {platform}: QA {qa_accuracy_percent:.2f}% ({qa_correct}/{qa_total}), "
            "evidence {evidence_hit_rate_percent:.2f}%, MRR {mrr:.4f}, model={answer_model}".format(**row),
            flush=True,
        )
    print(f"[compare] winner_by_qa_accuracy={winner['platform']}", flush=True)
    print(f"[compare] summary={summary_path}", flush=True)
    print(f"[compare] csv={csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
