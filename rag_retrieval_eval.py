import os
import json
import math
import random
import argparse
import numpy as np
import tqdm
import requests
import time

class BaseApi:
    session = requests.session()

    def request_send(self, method, url, **kwargs):
        res = self.session.request(method=method, url=url, **kwargs)
        return res

    def file_upload(self, indata, file_path):
        file_name = os.path.basename(file_path)
        files = {'file': (file_name, open(file_path, 'rb'), 'text/plain')}
        res = self.request_send(method='post', url='http://127.0.0.1:3000/api/fileupload', data=indata, files=files)
        return res.json()

    def delete_knowledge(self, payload):
        resp = self.request_send(method='delete', url='http://127.0.0.1:5001/api/knowledge', params=payload)
        return resp

def generate_unique_filename(base_name):
    timestamp = int(time.time())
    random_num = random.randint(1000, 9999)
    return f"{base_name}_{timestamp}_{random_num}.txt"

def flatten(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

def get_result_prompt(query):
    url = f"http://localhost:5001/api/search?query={query}&knowledgeLabel=1&reset=true"
    resp = requests.get(url)
    json_data = resp.json()
    return json_data

def parse_result_prompt(result_prompt):
    try:
        hits = result_prompt.get("hits", [])
        retrieved_docs = [hit.get("content", "") for hit in hits if hit.get("content") and isinstance(hit.get("content"), str)]
        return retrieved_docs
    except Exception:
        return []

def evaluate_retrieval(query, positive_docs, answer, retrieved_docs):
    num_positive = len(positive_docs)
    actual_k = len(retrieved_docs)
    if num_positive == 0 or actual_k == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "actual_k": actual_k
        }

    relevant = []
    for i, retrieved_doc in enumerate(retrieved_docs):
        is_answer_match = False
        for ans in answer:
            ans_str = str(ans).lower()
            doc_lower = retrieved_doc.lower()
            if ans_str in doc_lower:
                is_answer_match = True
                break
            if ans_str.isdigit():
                chinese_nums = {
                    '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
                    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
                }
                chinese_ans = ''.join(chinese_nums.get(d, d) for d in ans_str)
                if chinese_ans + '人' in doc_lower or chinese_ans in doc_lower:
                    is_answer_match = True
                    break
        relevant.append(1 if is_answer_match else 0)
    precision = sum(relevant) / actual_k if actual_k > 0 else 0.0
    recall = min(sum(relevant) / num_positive, 1.0) if num_positive > 0 else 0.0

    mrr = 0.0
    for idx, rel in enumerate(relevant, 1):
        if rel == 1:
            mrr = 1.0 / idx
            break

    dcg = 0.0
    for i, rel in enumerate(relevant, 1):
        dcg += rel / math.log2(i + 1)
    ideal_len = min(num_positive, actual_k)
    ideal_relevant = [1] * ideal_len + [0] * (actual_k - ideal_len)
    idcg = 0.0
    for i, rel in enumerate(ideal_relevant, 1):
        idcg += rel / math.log2(i + 1)
    ndcg = dcg / idcg if idcg > 0 else 0.0
    ndcg = min(ndcg, 1.0)

    return {
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "ndcg": ndcg,
        "actual_k": actual_k
    }

def clear_knowledge(knowledge_label):
    indata = {"knowledgeLabel": str(knowledge_label)}
    while True:
        resp = BaseApi().delete_knowledge(indata)
        try:
            if resp.status_code == 200 and "没有找到知识库" in resp.json().get('message', ''):
                break
        except Exception:
            break
        time.sleep(1)

def run_all_in_one(args):
    random.seed(None)
    instances = []
    with open(f'data/{args.dataset}.json', 'r', encoding='utf-8') as f:
        for line in f:
            instances.append(json.loads(line))

    # 合并所有positive和negative文档（兼容 zh_int 结构）
    all_docs = []
    id_to_positive = {}
    for instance in instances:
        if '_int' in args.dataset:
            processed_positive = list(flatten(instance['positive']))
        else:
            processed_positive = instance['positive']
        processed_negative = instance['negative']
        all_docs.extend(processed_positive)
        all_docs.extend(processed_negative)
        id_to_positive[instance['id']] = processed_positive

    merged_doc = "\n".join([doc.strip() for doc in all_docs if isinstance(doc, str) and doc.strip()])
    file_name = generate_unique_filename(f"all_in_one_{args.dataset}")
    file_path = os.path.join("temp_docs", file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(merged_doc)

    print("清空知识库...")
    clear_knowledge(knowledge_label=1)

    indata = {"knowledgeLabel": "1"}
    upload_resp = BaseApi().file_upload(indata, file_path)
    if upload_resp.get('status') != 200 or upload_resp.get('messages') != '上传成功':
        print("上传失败，退出")
        os.remove(file_path)
        return None, None

    results = []
    actual_ks = []
    upload_char_count = len(merged_doc)

    for instance in tqdm.tqdm(instances, desc="全量模式评测query"):
        query = instance['query']
        answer = instance['answer'] if isinstance(instance['answer'], list) else [instance['answer']]
        result_prompt = get_result_prompt(query)
        retrieved_docs = parse_result_prompt(result_prompt)
        positive_docs = id_to_positive[instance['id']]
        eval_metrics = evaluate_retrieval(query, positive_docs, answer, retrieved_docs)
        results.append({
            'id': instance['id'],
            'query': query,
            'positive_docs': positive_docs,
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'eval_metrics': eval_metrics
        })
        actual_ks.append(eval_metrics['actual_k'])

    print("测试完成，清空知识库...")
    clear_knowledge(knowledge_label=1)
    os.remove(file_path)

    resultpath = 'result-zh' if 'zh' in args.dataset else 'result-en'
    os.makedirs(resultpath, exist_ok=True)
    filename = os.path.join(resultpath,
                            f'retrieval_evaluation_{args.dataset}_allinone_{int(time.time())}.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    precision_list = [r['eval_metrics']['precision'] for r in results]
    recall_list = [r['eval_metrics']['recall'] for r in results]
    mrr_list = [r['eval_metrics']['mrr'] for r in results]
    ndcg_list = [r['eval_metrics']['ndcg'] for r in results]
    avg_k = np.mean(actual_ks)

    avg_metrics = {
        "precision": np.mean(precision_list),
        "precision_var": np.var(precision_list, ddof=1),
        "recall": np.mean(recall_list),
        "recall_var": np.var(recall_list, ddof=1),
        "mrr": np.mean(mrr_list),
        "mrr_var": np.var(mrr_list, ddof=1),
        "ndcg": np.mean(ndcg_list),
        "ndcg_var": np.var(ndcg_list, ddof=1),
        "avg_k": avg_k
    }

    print(f"本轮上传数据总字数 {upload_char_count} 字")
    return avg_metrics, upload_char_count

def process_single_instance(instance, noise_rate, passage_num, dataset):
    neg_num = math.ceil(passage_num * noise_rate)
    pos_num = passage_num - neg_num

    if '_int' in dataset:
        for i in instance['positive']:
            random.shuffle(i)
        docs = [i[0] for i in instance['positive']]
        if len(docs) < pos_num:
            maxnum = max(len(i) for i in instance['positive'])
            for i in range(1, maxnum):
                for j in instance['positive']:
                    if len(j) > i:
                        docs.append(j[i])
                        if len(docs) == pos_num:
                            break
                if len(docs) == pos_num:
                    break
        neg_num = passage_num - len(docs)
        if neg_num > 0:
            docs += instance['negative'][:neg_num]
    else:
        if noise_rate == 1:
            neg_num = passage_num
            pos_num = 0
        else:
            if neg_num > len(instance['negative']):
                neg_num = len(instance['negative'])
                pos_num = passage_num - neg_num
            if pos_num > len(instance['positive']):
                pos_num = len(instance['positive'])
                neg_num = passage_num - pos_num
        docs = instance['positive'][:pos_num] + instance['negative'][:neg_num]

    selected_positive_docs = instance['positive'][:pos_num]
    random.shuffle(docs)
    merged_doc = "\n".join(docs)
    file_name = generate_unique_filename(f"instance_{instance['id']}")
    file_path = os.path.join("temp_docs", file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(merged_doc)
    return file_path, selected_positive_docs

def run_single(args):
    random.seed(None)
    instances = []
    with open(f'data/{args.dataset}.json', 'r', encoding='utf-8') as f:
        for line in f:
            instances.append(json.loads(line))

    results = []
    upload_char_counts = []

    for instance in tqdm.tqdm(instances, desc="逐条处理query"):
        file_path, positive_docs = process_single_instance(
            instance, args.noise_rate, args.num, args.dataset
        )

        indata = {"knowledgeLabel": "1"}
        upload_resp = BaseApi().file_upload(indata, file_path)
        if upload_resp.get('status') != 200 or upload_resp.get('messages') != '上传成功':
            os.remove(file_path)
            continue

        query = instance['query']
        answer = instance['answer'] if isinstance(instance['answer'], list) else [instance['answer']]
        result_prompt = get_result_prompt(query)
        retrieved_docs = parse_result_prompt(result_prompt)
        eval_metrics = evaluate_retrieval(query, positive_docs, answer, retrieved_docs)

        with open(file_path, 'r', encoding='utf-8') as f:
            upload_char_count = len(f.read())
        upload_char_counts.append(upload_char_count)

        results.append({
            'id': instance['id'],
            'query': query,
            'positive_docs': positive_docs,
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'eval_metrics': eval_metrics
        })

        indata = {"knowledgeLabel": "1"}
        while True:
            resp = BaseApi().delete_knowledge(indata)
            try:
                if resp.status_code == 200 and "没有找到知识库 '1' 下的文档片段，无需删除。" in resp.json().get('message', ''):
                    break
            except Exception:
                break
            time.sleep(1)
        os.remove(file_path)

    resultpath = 'result-zh' if 'zh' in args.dataset else 'result-en'
    os.makedirs(resultpath, exist_ok=True)
    filename = os.path.join(resultpath,
                            f'retrieval_evaluation_{args.dataset}_noise{args.noise_rate}_passage{args.num}_{int(time.time())}.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    precision_list = [r['eval_metrics']['precision'] for r in results]
    recall_list = [r['eval_metrics']['recall'] for r in results]
    mrr_list = [r['eval_metrics']['mrr'] for r in results]
    ndcg_list = [r['eval_metrics']['ndcg'] for r in results]
    avg_k = np.mean([r['eval_metrics']['actual_k'] for r in results])

    avg_metrics = {
        "precision": np.mean(precision_list),
        "precision_var": np.var(precision_list, ddof=1),
        "recall": np.mean(recall_list),
        "recall_var": np.var(recall_list, ddof=1),
        "mrr": np.mean(mrr_list),
        "mrr_var": np.var(mrr_list, ddof=1),
        "ndcg": np.mean(ndcg_list),
        "ndcg_var": np.var(ndcg_list, ddof=1),
        "avg_k": avg_k
    }

    avg_char_count = np.mean(upload_char_counts)
    print(f"本轮平均每个query上传数据 {avg_char_count:.0f} 字")
    return avg_metrics, avg_char_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='zh_refine', help='evaluation dataset')
    parser.add_argument('--noise_rate', type=float, default=0.2, help='rate of noisy passages')
    parser.add_argument('--num', type=int, default=5, help='number of external passages')
    parser.add_argument('--all_in_one_upload', action='store_true', help='一次性上传全部文档并全局评测')
    args = parser.parse_args()

    print("开始运行检索评测流程")
    if args.all_in_one_upload:
        metrics, char_count = run_all_in_one(args)
    else:
        metrics, char_count = run_single(args)

    if metrics and char_count is not None:
        print("\n本次运行结果：")
        print(f"Precision: {metrics['precision']:.4f} (±{metrics['precision_var']:.4f})")
        print(f"Recall: {metrics['recall']:.4f} (±{metrics['recall_var']:.4f})")
        print(f"MRR: {metrics['mrr']:.4f} (±{metrics['mrr_var']:.4f})")
        print(f"NDCG: {metrics['ndcg']:.4f} (±{metrics['ndcg_var']:.4f})")
        print(f"平均检索文档数 k = {metrics['avg_k']:.2f}")
        if args.all_in_one_upload:
            print(f"\n本次上传数据总字数 {char_count:.0f} 字")
        else:
            print(f"\n本次每个query上传数据平均为 {char_count:.0f} 字")
    else:
        print("运行失败，未能计算指标")

if __name__ == '__main__':
    main()
