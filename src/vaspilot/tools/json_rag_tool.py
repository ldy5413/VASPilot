import json
import os
from typing import Any, Dict, List, Optional, Type, Set
from pathlib import Path

from crewai.tools.base_tool import BaseTool
from pydantic import BaseModel, Field, model_validator
import chromadb
from chromadb import EmbeddingFunction, Client, Collection, ClientAPI
from chromadb.utils import embedding_functions
import uuid


class JsonApproxSearchInput(BaseModel):
    """Input schema for JsonRagTool"""
    query: str = Field(description="text to query the docs")
    top_k: int = Field(default=10, description="number of results to return. The default value is 10.")


class JsonApproxSearch(BaseTool):
    """
    Use RAG technology to search for relevant information from the JSON knowledge base.
    JSON format: {tag_name: {default_value, description, detailed_description, related_tags}}
    """
    
    name: str = "approximate_search_tool"
    description: str = (
        "Use RAG technology to search for relevant information from the JSON knowledge base."
        "Can search for the most relevant configuration items and return short details."
    )
    args_schema: Type[BaseModel] = JsonApproxSearchInput
    
    # 添加embedding_function作为字段
    embedding_function: EmbeddingFunction = Field(description="Embedding function for ChromaDB")
    
    # 声明实例属性类型，添加默认值
    client: Optional[ClientAPI] = Field(default=None)
    collection: Optional[Collection] = Field(default=None)
    added_files: Set[str] = Field(default_factory=set)
    
    @model_validator(mode='after')
    def initialize_components(self) -> 'JsonApproxSearch':
        """Pydantic v2风格的初始化方法"""
        # 初始化ChromaDB客户端
        self.client = chromadb.Client()
        
        # 创建或获取集合
        self.collection = self.client.get_or_create_collection(
            name="json_knowledge_base",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function
        )
        
        # 用于跟踪已添加的文件
        if self.added_files is None:
            self.added_files = set()
            
        return self
    
    def add(self, json_file_path: str) -> None:
        """
        添加JSON文件到知识库
        
        Args:
            json_file_path: JSON文件的路径
        """
        json_path = Path(json_file_path)
        
        # 检查文件是否存在
        if not json_path.exists():
            raise FileNotFoundError(f"文件不存在: {json_file_path}")
        
        # 检查是否已添加
        if str(json_path.absolute()) in self.added_files:
            print(f"文件已在知识库中: {json_file_path}")
            return
        
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 准备数据用于向量化
        documents = []
        metadatas = []
        ids = []
        
        for tag_name, tag_info in data.items():
            # 构建用于向量化的文本
            text_parts = [
                f"Tag name: {tag_name}",
                f"Default value: {tag_info.get('default_value', '')}",
                f"Description: {tag_info.get('description', '')}",
                f"Detailed description: {tag_info.get('detailed_description', '')}",
                f"Related tags: {', '.join(tag_info.get('related_tags', []))}"
            ]
            document_text = "\n".join(text_parts)
            
            # 准备元数据
            metadata = {
                "tag_name": tag_name,
                "default_value": tag_info.get('default_value', ''),
                "description": tag_info.get('description', ''),
                "source_file": str(json_path.absolute()),
                "related_tags": json.dumps(tag_info.get('related_tags', [])),
            }
            
            documents.append(document_text)
            metadatas.append(metadata)
            ids.append(f"{tag_name}_{str(uuid.uuid4())[:8]}")
        
        # 添加到向量数据库
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"成功添加 {len(documents)} 个标签到知识库: {json_file_path}")
        
        # 记录已添加的文件
        self.added_files.add(str(json_path.absolute()))
    
    def _run(self, query: str, top_k: int = 10) -> dict:
        """
        执行RAG查询
        
        Args:
            query: 查询文本
            top_k: 返回结果的数量
            
        Returns:
            查询结果的字典
        """
        # 检查知识库是否为空
        if self.collection.count() == 0:
            return "知识库为空，请先使用 add() 方法添加JSON文件。"
        
        # 执行查询
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # 格式化结果
        if not results['documents'][0]:
            return "未找到相关的标签信息。"
        
        response = {}
        
        for i, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0]), 1):
            tag_name = metadata['tag_name']
            description = metadata['description']
            
            response[tag_name] = {
                "description": description,
                "default_value": metadata['default_value'],
                "related_tags": metadata['related_tags'],
                "score": f"{1 - distance:.3f}"
            }
        
        return response
    
    def clear_knowledge_base(self) -> None:
        """清空知识库"""
        self.client.delete_collection(name="json_knowledge_base")
        self.collection = self.client.create_collection(
            name="json_knowledge_base",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function
        )
        self.added_files.clear()
        print("知识库已清空。")
    
    def list_added_files(self) -> List[str]:
        """列出已添加到知识库的文件"""
        return list(self.added_files)

class JsonStrictSearchInput(BaseModel):
    tag_name: str = Field(description="the exact tag_name to query details.")

class JsonStrictSearch(BaseTool):
    """
    Tool to query detailed descriptions
    """
    name: str = "strict_search_tool"
    description: str = "Tool to query detailed descriptions. The detailed description is long, only query the most important tags."
    args_schema: Type[BaseModel] = JsonStrictSearchInput
    
    # 声明实例属性类型
    data_dict: Dict[str, Any] = Field(default_factory=dict)

    def add(self, json_file_path: str) -> None:
        """
        添加JSON文件到知识库
        
        Args:
            json_file_path: JSON文件的路径
        """
        json_path = Path(json_file_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for tag_name, tag_info in data.items():
            self.data_dict[tag_name] = tag_info

    def _run(self, tag_name: str) -> str:
        if self.data_dict.get(tag_name, None) is not None:
            return self.data_dict[tag_name]
        else:
            return f"No detailed description found for tag_name: {tag_name}"