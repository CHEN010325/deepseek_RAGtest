import argparse
import json
import math
import os
import random
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import requests
import tqdm


API_BASE = os.environ.get("RAG_API_BASE", "http://127.0.0.1:3335").rstrip("/")
KNOWLEDGE_LABEL = os.environ.get("RAG_KB_LABEL", "1")
USE_ADAPTIVE_RAG = os.environ.get("RAG_USE_ADAPTIVE", "true").lower() in ("1", "true", "yes", "on")
MEILI_BASE = os.environ.get("MEILI_DEV_BASE", "http://127.0.0.1:7775").rstrip("/")
MEILI_KEY = os.environ.get("MEILI_API_KEY", "qaz0913cde350odxs")
SEARCH_MODE = os.environ.get("RAG_SEARCH_MODE", "").strip()
SEARCH_MODEL = os.environ.get("RAG_SEARCH_MODEL", "").strip()
ANSWER_MODEL = os.environ.get("RAG_ANSWER_MODEL", "qwen3.5:4b")
ANSWER_BACKEND = os.environ.get("RAG_ANSWER_BACKEND", "mimo").lower()
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_NUM_CTX = int(os.environ.get("RAG_OLLAMA_NUM_CTX", "32768"))
OLLAMA_TIMEOUT = int(os.environ.get("RAG_OLLAMA_TIMEOUT", "900"))
OLLAMA_NO_THINK = os.environ.get("RAG_OLLAMA_NO_THINK", "true").lower() in ("1", "true", "yes", "on")
OLLAMA_NUM_PREDICT = int(os.environ.get("RAG_OLLAMA_NUM_PREDICT", "512"))
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "").strip()
MIMO_BASE = os.environ.get("MIMO_BASE", "https://api.xiaomimimo.com/v1").rstrip("/")
EVAL_DOC_IDS = []

DATASET_CORPUS = {
    "zh_refine": "merged_all_docs.txt",
    "zh_int": "merged_all_docs2.txt",
    "zh_int_clean": "merged_all_docs2.txt",
}


def flatten(items):
    for item in items:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


def load_dataset(dataset_name, limit=0):
    data_path = Path("data") / f"{dataset_name}.json"
    rows = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit and limit > 0:
        rows = rows[:limit]
    return rows


def filter_dataset_by_ids(rows, ids_text):
    ids = [int(x.strip()) for x in str(ids_text or "").split(",") if x.strip()]
    if not ids:
        return rows
    row_ids = {int(row.get("id", -1)) for row in rows}
    missing = [row_id for row_id in ids if row_id not in row_ids]
    if missing:
        raise ValueError(f"--ids contains unknown ids: {missing}")
    wanted = set(ids)
    return [row for row in rows if int(row.get("id", -1)) in wanted]


def clear_knowledge():
    resp = requests.delete(
        f"{MEILI_BASE}/indexes/kb_{KNOWLEDGE_LABEL}/documents",
        headers={"Authorization": f"Bearer {MEILI_KEY}"},
        timeout=120,
    )
    if resp.status_code == 404:
        return
    if resp.status_code not in (200, 202):
        print(f"[评测] 清空知识库失败：HTTP {resp.status_code} {resp.text[:200]}")
        return
    task_uid = resp.json().get("taskUid")
    if task_uid is not None:
        wait_meili_task(task_uid)
    time.sleep(1)


def wait_meili_task(task_uid, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(
            f"{MEILI_BASE}/tasks/{task_uid}",
            headers={"Authorization": f"Bearer {MEILI_KEY}"},
            timeout=30,
        )
        if resp.status_code == 404:
            return
        resp.raise_for_status()
        task = resp.json()
        status = task.get("status")
        if status == "succeeded":
            return
        if status in ("failed", "canceled"):
            raise RuntimeError(f"Meilisearch task {task_uid} failed: {task}")
        time.sleep(0.5)
    raise TimeoutError(f"Meilisearch task {task_uid} timeout")


def upload_text_file(file_path):
    with open(file_path, "rb") as f:
        resp = requests.post(
            f"{API_BASE}/api/files/upload",
            data={
                "knowledgeLabel": KNOWLEDGE_LABEL,
                "waitForProcessing": "true",
                "fastParsing": "true",
            },
            files={"file": (Path(file_path).name, f, "text/plain")},
            timeout=600,
        )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"上传失败：{payload}")
    return payload


def search(query):
    payload = {
        "query": query,
        "knowledgeLabel": KNOWLEDGE_LABEL,
        "reset": True,
        "useAdaptiveRag": USE_ADAPTIVE_RAG,
        "docIds": EVAL_DOC_IDS,
    }
    if SEARCH_MODE:
        payload["mode"] = SEARCH_MODE
    if SEARCH_MODEL:
        payload["model"] = SEARCH_MODEL

    resp = requests.post(
        f"{API_BASE}/api/search",
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def parse_retrieved_docs(search_result):
    hits = search_result.get("hits", [])
    return [
        hit.get("content", "")
        for hit in hits
        if isinstance(hit.get("content"), str) and hit.get("content")
    ]


def parse_sse_text(text):
    output = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload:
            continue
        try:
            data = json.loads(payload)
        except Exception:
            continue
        if isinstance(data, dict):
            content = data.get("content") or data.get("delta") or data.get("text") or ""
            if isinstance(content, str):
                output.append(content)
            message = data.get("message")
            if isinstance(message, str) and data.get("type") not in ("error",):
                output.append(message)
    return "".join(output).strip()


def answer_with_app_chat(prompt):
    if ANSWER_BACKEND == "mimo":
        return answer_with_mimo(prompt)
    if ANSWER_BACKEND == "ollama":
        return answer_with_ollama(prompt)

    resp = requests.post(
        f"{API_BASE}/api/chat",
        json={
            "prompt": prompt,
            "model": ANSWER_MODEL,
            "history": [],
        },
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return parse_sse_text(resp.text)


def answer_with_mimo(prompt):
    if not MIMO_API_KEY:
        raise RuntimeError("MIMO_API_KEY is required when RAG_ANSWER_BACKEND=mimo")
    resp = requests.post(
        f"{MIMO_BASE}/chat/completions",
        headers={
            "api-key": MIMO_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "model": ANSWER_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个严谨的中文问答助手。只根据用户提供的上下文回答，答案尽量简短，直接给出结论。",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "max_completion_tokens": 2048,
            "temperature": 0,
            "top_p": 0.95,
            "stream": False,
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    message = data.get("choices", [{}])[0].get("message", {}) or {}
    return strip_thinking(message.get("content") or message.get("reasoning_content") or "")


def answer_with_ollama(prompt):
    ollama_prompt = f"{prompt}\n\n/no_think" if OLLAMA_NO_THINK else prompt
    options = {
        "temperature": 0,
        "num_ctx": OLLAMA_NUM_CTX,
    }
    if OLLAMA_NUM_PREDICT > 0:
        options["num_predict"] = OLLAMA_NUM_PREDICT

    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": ANSWER_MODEL,
            "messages": [
                {"role": "user", "content": ollama_prompt},
            ],
            "stream": False,
            "think": not OLLAMA_NO_THINK,
            "options": options,
        },
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    message = resp.json().get("message", {}) or {}
    text = message.get("content", "")
    return strip_thinking(text)


def strip_thinking(text):
    text = str(text or "")
    while "<think>" in text and "</think>" in text:
        before, rest = text.split("<think>", 1)
        _, after = rest.split("</think>", 1)
        text = before + after
    return text.strip()


def normalize_for_match(text):
    text = unicodedata.normalize("NFKC", str(text or "").lower())
    replacements = {
        "杜琪峯": "杜琪峰",
        "西敏寺": "威斯敏斯特",
        "westminsterabbey": "威斯敏斯特",
        "westminster": "威斯敏斯特",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"(\d+)\.0+(%)", r"\1\2", text)
    text = re.sub(r"[,\s，。．、:：;；!！?？（）()【】\[\]《》“”\"'`*_~～—-]", "", text)
    return text


def check_answer(prediction, answers):
    pred = normalize_for_match(prediction)
    answer_list = answers if isinstance(answers, list) else [answers]
    labels = []
    for answer in answer_list:
        if isinstance(answer, list):
            labels.append(1 if any(normalize_for_match(x) in pred for x in answer) else 0)
        else:
            labels.append(1 if normalize_for_match(answer) in pred else 0)
    return labels


def is_answer_correct(prediction, answers):
    labels = check_answer(prediction, answers)
    return bool(labels) and 0 not in labels and 1 in labels


def evaluate_retrieval(answers, positive_docs, retrieved_docs):
    answers = answers if isinstance(answers, list) else [answers]
    answer_terms = list(flatten(answers))
    positive_count = max(1, len(positive_docs))
    actual_k = len(retrieved_docs)
    if actual_k == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "actual_k": 0,
        }

    relevant = []
    for doc in retrieved_docs:
        doc_norm = normalize_for_match(doc)
        matched = any(normalize_for_match(answer) in doc_norm for answer in answer_terms)
        relevant.append(1 if matched else 0)

    precision = sum(relevant) / actual_k
    recall = min(sum(relevant) / positive_count, 1.0)

    mrr = 0.0
    for idx, rel in enumerate(relevant, 1):
        if rel:
            mrr = 1.0 / idx
            break

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevant))
    ideal_len = min(positive_count, actual_k)
    ideal = [1] * ideal_len + [0] * (actual_k - ideal_len)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    ndcg = dcg / idcg if idcg else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "ndcg": min(ndcg, 1.0),
        "actual_k": actual_k,
    }


def build_all_in_one_docs(instances, dataset):
    all_docs = []
    id_to_positive = {}
    for row in instances:
        positives = list(flatten(row["positive"])) if "_int" in dataset else row["positive"]
        negatives = row.get("negative", [])
        all_docs.extend(positives)
        all_docs.extend(negatives)
        id_to_positive[row["id"]] = positives
    merged = "\n".join(str(doc).strip() for doc in all_docs if isinstance(doc, str) and doc.strip())
    return merged, id_to_positive


def build_eval_corpus(instances, dataset):
    corpus_file = DATASET_CORPUS.get(dataset)
    if corpus_file:
        path = Path(corpus_file)
        if not path.exists():
            raise FileNotFoundError(f"语料文件不存在：{path.resolve()}")
        return path.read_text(encoding="utf-8"), {
            row["id"]: (list(flatten(row["positive"])) if "_int" in dataset else row["positive"])
            for row in instances
        }, corpus_file

    merged_doc, id_to_positive = build_all_in_one_docs(instances, dataset)
    return merged_doc, id_to_positive, "generated"


def summarize(results):
    metrics = [r["eval_metrics"] for r in results]
    qa_results = [r for r in results if "answer_correct" in r]
    correct_count = sum(1 for r in qa_results if r.get("answer_correct"))
    return {
        "precision": float(np.mean([m["precision"] for m in metrics])) if metrics else 0.0,
        "recall": float(np.mean([m["recall"] for m in metrics])) if metrics else 0.0,
        "mrr": float(np.mean([m["mrr"] for m in metrics])) if metrics else 0.0,
        "ndcg": float(np.mean([m["ndcg"] for m in metrics])) if metrics else 0.0,
        "avg_k": float(np.mean([m["actual_k"] for m in metrics])) if metrics else 0.0,
        "qa_correct": correct_count,
        "qa_total": len(qa_results),
        "qa_accuracy": correct_count / len(qa_results) if qa_results else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="zh_int")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条；0 表示全量")
    parser.add_argument("--ids", default="", help="只跑指定行 id，多个用逗号分隔")
    parser.add_argument("--mode", choices=["retrieval", "qa"], default="retrieval")
    parser.add_argument("--system-name", default="current")
    parser.add_argument("--skip-upload", action="store_true", help="复用已上传文档，不清空、不上传")
    parser.add_argument("--doc-ids", default="", help="复用文档名，多个用逗号分隔")
    parser.add_argument("--keep-kb", action="store_true", help="结束后不清空测试知识库")
    args = parser.parse_args()

    random.seed(42)
    instances = filter_dataset_by_ids(load_dataset(args.dataset, args.limit), args.ids)
    print(f"[评测] API：{API_BASE}")
    print(f"[评测] 知识库：{KNOWLEDGE_LABEL}")
    print(f"[评测] 数据集：{args.dataset}，问题数：{len(instances)}，模式：{args.mode}")

    merged_doc, id_to_positive, corpus_source = build_eval_corpus(instances, args.dataset)
    temp_file = None
    if not args.skip_upload:
        temp_dir = Path("temp_docs")
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / f"current_eval_{args.dataset}_{int(time.time())}.txt"
        temp_file.write_text(merged_doc, encoding="utf-8")

    results = []
    try:
        if args.skip_upload:
            doc_ids = [x.strip() for x in args.doc_ids.split(",") if x.strip()]
            if not doc_ids:
                raise ValueError("--skip-upload requires --doc-ids")
            EVAL_DOC_IDS[:] = doc_ids
            print(f"[评测] 复用测试文档：{', '.join(EVAL_DOC_IDS)}")
        else:
            print("[评测] 清空测试知识库")
            clear_knowledge()
            print(f"[评测] 上传测试文档：{corpus_source}，{len(merged_doc)} 字")
            upload_payload = upload_text_file(temp_file)
            EVAL_DOC_IDS[:] = [str(x) for x in upload_payload.get("files", []) if str(x).strip()]
            if not EVAL_DOC_IDS:
                EVAL_DOC_IDS.append(temp_file.name)
            print(f"[评测] 测试文档：{', '.join(EVAL_DOC_IDS)}")

        for row in tqdm.tqdm(instances, desc="检索评测"):
            query = row["query"]
            answers = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]
            search_result = search(query)
            retrieved_docs = parse_retrieved_docs(search_result)
            metrics = evaluate_retrieval(answers, id_to_positive[row["id"]], retrieved_docs)
            item = {
                "id": row["id"],
                "query": query,
                "answer": answers,
                "retrieved_count": len(retrieved_docs),
                "eval_metrics": metrics,
            }
            if args.mode == "qa":
                result_prompt = search_result.get("result_prompt", "")
                prediction = answer_with_app_chat(result_prompt)
                item["prediction"] = prediction
                item["answer_correct"] = is_answer_correct(prediction, answers)
            results.append(item)
    finally:
        if not args.keep_kb and not args.skip_upload:
            print("[评测] 清空测试知识库")
            clear_knowledge()
        if temp_file and temp_file.exists():
            temp_file.unlink()

    out_dir = Path("result-zh" if "zh" in args.dataset else "result-en")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{args.system_name}_{args.mode}_eval_{args.dataset}_{int(time.time())}.json"
    report = {
        "api_base": API_BASE,
        "knowledge_label": KNOWLEDGE_LABEL,
        "system_name": args.system_name,
        "mode": args.mode,
        "answer_model": ANSWER_MODEL if args.mode == "qa" else None,
        "answer_backend": ANSWER_BACKEND if args.mode == "qa" else None,
        "dataset": args.dataset,
        "corpus_source": corpus_source,
        "query_count": len(instances),
        "summary": summarize(results),
        "results": results,
    }
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    print("[评测] 完成")
    print(f"[评测] Precision：{summary['precision']:.4f}")
    print(f"[评测] Recall：{summary['recall']:.4f}")
    print(f"[评测] MRR：{summary['mrr']:.4f}")
    print(f"[评测] NDCG：{summary['ndcg']:.4f}")
    print(f"[评测] 平均入模块数：{summary['avg_k']:.2f}")
    if args.mode == "qa":
        print(f"[评测] 问答准确率：{summary['qa_accuracy'] * 100:.2f}% ({summary['qa_correct']}/{summary['qa_total']})")
    print(f"[评测] 报告：{out_file.resolve()}")


if __name__ == "__main__":
    main()
