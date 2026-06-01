import os
import json
import math
import random
import argparse
import numpy as np
import tqdm
import yaml
import requests
import time
import shutil

from models import *


# BaseApi 类，用于文件上传和删除
class BaseApi:
    session = requests.session()

    def request_send(self, method, url, **kwargs):
        res = self.session.request(method=method, url=url, **kwargs)
        return res

    def file_upload(self, indata, file_path):
        file_name = file_path.split('\\')[-1]
        files = {'file': (file_name, open(file_path, 'rb'), 'text/plain')}
        res = self.request_send(method='post', url='http://127.0.0.1:3000/api/fileupload', data=indata, files=files)
        return res.json()

    def delete_file(self, indata):
        params = {'doc_id': indata}
        res = self.request_send(method='delete', url='http://127.0.0.1:5001/api/files', params=params)
        return res.json()


# 生成唯一文件名
def generate_unique_filename(base_name):
    timestamp = int(time.time())
    random_num = random.randint(1000, 9999)
    return f"{base_name}_{timestamp}_{random_num}.txt"


def process_data(instance, noise_rate, passage_num, filename, correct_rate=0):
    """
    处理数据，生成正负样本文档列表，并合并为一个文件
    """
    query = instance['query']
    ans = instance['answer']

    neg_num = math.ceil(passage_num * noise_rate)
    pos_num = passage_num - neg_num

    if '_int' in filename:
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
    elif '_fact' in filename:
        correct_num = math.ceil(passage_num * correct_rate)
        pos_num = passage_num - neg_num - correct_num
        indexs = list(range(len(instance['positive'])))
        selected = random.sample(indexs, min(len(indexs), pos_num))
        docs = [instance['positive_wrong'][i] for i in selected]
        remain = [i for i in indexs if i not in selected]
        if correct_num > 0 and len(remain) > 0:
            docs += [instance['positive'][i] for i in random.sample(remain, min(len(remain), correct_num))]
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
            elif pos_num > len(instance['positive']):
                pos_num = len(instance['positive'])
                neg_num = passage_num - pos_num
        docs = instance['positive'][:pos_num] + instance['negative'][:neg_num]

    random.shuffle(docs)

    # 合并所有文档为一个字符串
    merged_doc = "\n".join(docs)

    # 生成唯一的文件名
    file_name = generate_unique_filename("merged_doc")
    file_path = os.path.join("temp_docs", file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 保存为单个 TXT 文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(merged_doc)

    return query, ans, docs, file_path


def get_result_prompt(query):
    """
    Use the query to request an interface and obtain result_prompt.
    """
    url = f"http://localhost:5001/api/search?query={query}&knowledgeLabel=1&reset=true"
    resp = requests.get(url)
    return resp.json().get("result_prompt")


def check_answer(prediction, ground_truth):
    """
    Check the match between the predicted answer and the ground truth.
    """
    prediction = prediction.lower()
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]
    labels = []
    for gt in ground_truth:
        flag = True
        if isinstance(gt, list):
            flag = False
            gt = [i.lower() for i in gt]
            for i in gt:
                if i in prediction:
                    flag = True
                    break
        else:
            if gt.lower() not in prediction:
                flag = False
        labels.append(int(flag))
    return labels


def predict(query, ground_truth, model, dataset, file_path):
    """
    Generate model predictions and assign labels using result_prompt.
    """
    # 上传文件
    doc_id = None
    upload_resp = None
    if file_path:
        indata = {"knowledgeLabel": "1"}
        try:
            upload_resp = BaseApi().file_upload(indata, file_path)
            if upload_resp.get('status') == 200 and upload_resp.get('messages') == '上传成功':
                doc_id = upload_resp['data']['data'][0]['data'][0]['doc_index']
        except Exception:
            pass

    # 获取 result_prompt 并生成预测
    result_prompt = get_result_prompt(query)
    prediction = model.generate(result_prompt)

    if 'zh' in dataset:
        prediction = ''.join(prediction.split())

    if '信息不足' in prediction or 'insufficient information' in prediction:
        labels = [-1]
    else:
        labels = check_answer(prediction, ground_truth)

    factlabel = int('事实性错误' in prediction or 'factual errors' in prediction)

    return labels, prediction, factlabel, doc_id, upload_resp


def safe_filename(s):
    """
    Handle special characters in filenames.
    """
    return str(s).replace('/', '_')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str, default='chatgpt', help='model name')
    parser.add_argument('--dataset', type=str, default='en', help='evaluation dataset',
                        choices=['en', 'zh', 'zh_refine','en_int', 'zh_int', 'en_fact', 'zh_fact'])
    parser.add_argument('--api_key', type=str, default='api_key', help='api key of chatgpt')
    parser.add_argument('--plm', type=str, default='THUDM/chatglm-6b', help='name of plm')
    parser.add_argument('--url', type=str, default='https://api.openai.com/v1/completions', help='url of chatgpt')
    parser.add_argument('--temp', type=float, default=0.7, help='temperature')
    parser.add_argument('--noise_rate', type=float, default=0.0, help='rate of noisy passages')
    parser.add_argument('--correct_rate', type=float, default=0.0, help='rate of correct passages')
    parser.add_argument('--passage_num', type=int, default=5, help='number of external passages')
    parser.add_argument('--factchecking', type=bool, default=False, help='whether to fact checking')
    args = parser.parse_args()

    modelname = args.modelname
    temperature = args.temp
    noise_rate = args.noise_rate
    passage_num = args.passage_num

    # 加载数据
    instances = []
    with open(f'data/{args.dataset}.json', 'r', encoding='utf-8') as f:
        for line in f:
            instances.append(json.loads(line))

    # 结果路径处理
    resultpath = 'result-en' if 'en' in args.dataset else 'result-zh'
    resultpath = os.path.abspath(resultpath)
    os.makedirs(resultpath, exist_ok=True)

    # 模型初始化
    model = None
    if modelname == 'chatgpt':
        model = OpenAIAPIModel(api_key=args.api_key, url=args.url)
    elif 'Llama-2' in modelname:
        model = LLama2(plm=args.plm)
    elif 'chatglm' in modelname:
        model = ChatglmModel(plm=args.plm)
    elif 'moss' in modelname:
        model = Moss(plm=args.plm)
    elif 'vicuna' in modelname:
        model = Vicuna(plm=args.plm)
    elif 'Qwen' in modelname:
        model = Qwen(plm=args.plm)
    elif 'Baichuan' in modelname:
        model = Baichuan(plm=args.plm)
    elif 'WizardLM' in modelname:
        model = WizardLM(plm=args.plm)
    elif 'BELLE' in modelname:
        model = BELLE(plm=args.plm)
    elif modelname.startswith('deepseek-r1'):
        model = DeepSeekModel(model_name=modelname)
    elif modelname.startswith('qwen3'):
        model = Qwen3(model_name=modelname)

    # 文件名处理
    modelname_clean = safe_filename(modelname)
    temperature_str = safe_filename(temperature)
    noise_rate_str = safe_filename(noise_rate)
    passage_num_str = safe_filename(passage_num)
    correct_rate_str = safe_filename(args.correct_rate)

    filename = os.path.join(
        resultpath,
        f'prediction_{args.dataset}_{modelname_clean}_temp{temperature_str}_noise{noise_rate_str}_passage{passage_num_str}_correct{correct_rate_str}.json'
    )
    useddata = {}
    if os.path.exists(filename):
        with open(filename, encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                useddata[data['id']] = data

    results = []
    failed_deletions = []  # 记录删除失败的 doc_id
    with open(filename, 'w', encoding='utf-8') as f:
        for instance in tqdm.tqdm(instances):
            if (instance['id'] in useddata and
                    instance['query'] == useddata[instance['id']]['query'] and
                    instance['answer'] == useddata[instance['id']]['ans']):
                results.append(useddata[instance['id']])
                f.write(json.dumps(useddata[instance['id']], ensure_ascii=False) + '\n')
                continue
            try:
                random.seed(2333)
                if passage_num == 0:
                    query = instance['query']
                    ans = instance['answer']
                    docs = []
                    file_path = None
                    label, prediction, factlabel, doc_id, upload_resp = predict(query, ans, model, args.dataset,
                                                                                file_path)
                else:
                    query, ans, docs, file_path = process_data(instance, noise_rate, passage_num, args.dataset,
                                                               args.correct_rate)
                    label, prediction, factlabel, doc_id, upload_resp = predict(query, ans, model, args.dataset,
                                                                                file_path)

                # 删除上传的文件
                if doc_id:
                    try:
                        resp_delete = BaseApi().delete_file(doc_id)
                        deleted_segments = len(resp_delete.get('deleted_ids', []))
                        uploaded_segments = len(
                            upload_resp['data']['data'][0]['data']) if upload_resp and 'data' in upload_resp else 0
                        if not ('已删除' in resp_delete.get('message', '') and deleted_segments > 0):
                            failed_deletions.append(doc_id)
                        elif deleted_segments != uploaded_segments and uploaded_segments != 0:
                            failed_deletions.append(
                                f"{doc_id} (预期 {uploaded_segments} 个片段，实际删除 {deleted_segments} 个)")
                    except Exception:
                        failed_deletions.append(doc_id)

                # 删除临时文件
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)

                newinstance = {
                    'id': instance['id'],
                    'query': query,
                    'ans': ans,
                    'label': label,
                    'prediction': prediction,
                    'docs': docs if passage_num > 0 else [],
                    'noise_rate': noise_rate,
                    'factlabel': factlabel
                }
                results.append(newinstance)
                f.write(json.dumps(newinstance, ensure_ascii=False) + '\n')
            except Exception:
                continue

    # 保存删除失败的 doc_id
    if failed_deletions:
        with open(os.path.join(resultpath, 'failed_deletions.log'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(failed_deletions))

    # 清理临时文件目录
    shutil.rmtree("temp_docs", ignore_errors=True)

    # 统计与评估
    tt = 0
    for i in results:
        label = i['label']
        if noise_rate == 1 and label[0] == -1:
            tt += 1
        elif 0 not in label and 1 in label:
            tt += 1
    print(f"模型：{modelname}，数据集：{args.dataset}，正确率：{tt / len(results) * 100:.2f}%")
    scores = {
        'all_rate': tt / len(results),
        'noise_rate': noise_rate,
        'tt': tt,
        'nums': len(results),
    }
    if '_fact' in args.dataset:
        fact_tt = 0
        correct_tt = 0
        for i in results:
            if i['factlabel'] == 1:
                fact_tt += 1
                if 0 not in i['label']:
                    correct_tt += 1
        scores['fact_check_rate'] = fact_tt / len(results)
        scores['correct_rate'] = correct_tt / fact_tt if fact_tt > 0 else 0
        scores['fact_tt'] = fact_tt
        scores['correct_tt'] = correct_tt

    # 结果保存
    result_json_name = f'prediction_{args.dataset}_{modelname_clean}_temp{temperature_str}_noise{noise_rate_str}_passage{passage_num_str}_correct{correct_rate_str}_result.json'
    result_json_path = os.path.join(resultpath, result_json_name)
    try:
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)
    except Exception:
        pass


if __name__ == '__main__':
    main()