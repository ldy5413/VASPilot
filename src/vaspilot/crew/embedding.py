import requests
from chromadb import Documents, Embeddings, EmbeddingFunction
from typing import cast
class LocalAPIEmbedder(EmbeddingFunction):
    def __init__(self, url: str = "http://172.16.8.24:8003/v1/embeddings", 
                 model_id: str = "BAAI/bge-m3",
                 api_key: str = "EMPTY"):
        self.url = url
        self.model_id = model_id
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def __call__(self, input: Documents) -> Embeddings:
        """将文本列表发送到远程 embedding API 进行处理"""
        payload = {
            "input": input,
            "model": self.model_id
        }

        response = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            timeout=30  # 设置超时时间（根据实际情况调整）
        )

        # 处理 API 响应
        if response.status_code != 200:
            raise Exception(f"API调用失败: {response.text}")

        # 根据实际API返回结构提取结果
        results = response.json()
        
        # 假设返回的数据结构：
        # {
        #   "embeddings": [[...], [...], ...]
        # }
        # 如果实际结构不同需要在此修改
        embeddings = results.get("data")
        
        sorted_embeddings = sorted(
                embeddings, key=lambda e: e["index"]  # type: ignore
            )

        return cast(
                Embeddings, [result["embedding"] for result in sorted_embeddings]
            )
    