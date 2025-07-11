import os

datasets = ["zh_refine", "zh_int"]
models = ["qwen3:0.6b", "qwen3:1.7b", "qwen3:4b","qwen3:8b"]
noise_rates = [0.0,0.2,0.4, 0.6, 0.8,1.0]
passage_num = 5

for dataset in datasets:
    for model in models:
        # 如果是 zh_int，只运行 noise_rate 为 0.0 的
        if dataset == "zh_int":
            noise_rate_list = [0.0]
        else:
            noise_rate_list = noise_rates

        for noise_rate in noise_rate_list:
            cmd = f"python new_test.py --dataset {dataset} --modelname {model} --noise_rate {noise_rate} --passage_num {passage_num}"
            print(f"Running: {cmd}")
            os.system(cmd)
