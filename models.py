from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch



class DeepSeekModel:
    def __init__(self, model_name: str = 'deepseek-r1:1.5b', plm: str = None) -> None:
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        print(f"Initialized Ollama model: {self.model_name}")
        try:
            response = requests.get("http://localhost:11434")
            if response.status_code != 200:
                raise Exception("Ollama service is not running or accessible")
        except requests.ConnectionError:
            raise Exception("Failed to connect to Ollama at http://localhost:11434. Ensure Ollama is running.")

    def generate(self, text: str, system: str = "") -> str:
        """
        Generate text using the Ollama model with its default parameters.
        """
        prompt = text
        if system:
            prompt = f"{system}\n\n{text}"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return ""


class Qwen3:
    def __init__(self, model_name: str = 'qwen3:8b') -> None:
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/chat"  # 改为chat接口
        print(f"Initialized Qwen3 model: {self.model_name}")

        # 强化服务检查（中文提示）
        try:
            if requests.get("http://localhost:11434").status_code != 200:
                raise RuntimeError("Ollama服务未运行！请执行：\n1. ollama serve\n2. ollama pull qwen3:8b")
        except requests.ConnectionError:
            raise ConnectionError("无法连接Ollama服务，请确认：\n- 已安装Ollama\n- 终端执行了ollama serve")

    def generate(self, text: str, system: str = "") -> str:
        """保持与DeepSeek完全兼容的接口"""
        # 构建消息体
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": text + " /think"})  # 默认启用思考模式

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            #"options": {"num_ctx": 8192}  # 显式设置上下文窗口
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()['message']['content'].strip()
        except requests.RequestException as e:
            return ""
        except KeyError:
            print("响应格式错误，请检查模型是否加载成功")
            return ""