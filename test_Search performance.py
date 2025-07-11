import os

noise_rates = [0.0, 0.2, 0.4, 0.6, 0.8]

for rate in noise_rates:
    cmd = f"python rag_retrieval_eval.py --dataset zh_int --noise_rate {rate} --num 5"
    os.system(cmd)

python rag_retrieval_eval.py --dataset zh_refine --all_in_one_upload
python rag_retrieval_eval.py --dataset zh_int --all_in_one_upload