# cache/semantic_cache.py
import pickle
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import faiss
from langchain_huggingface import HuggingFaceEmbeddings


class SemanticCache:
    """基于FAISS的语义缓存"""

    def __init__(self,
                 embedding_model,
                 cache_dir: str = "data/cache/semantic",
                 similarity_threshold: float = 0.85):
        """
        初始化语义缓存

        Args:
            embedding_model: 嵌入模型
            cache_dir: 缓存目录
            similarity_threshold: 相似度阈值（0-1）
        """
        self.embedding_model = embedding_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold

        # 存储缓存数据
        self.queries = []  # 原始查询列表
        self.responses = []  # 对应响应列表
        self.embeddings = []  # 查询向量列表

        # FAISS索引
        self.index = None
        self.dimension = None

        # 加载已有缓存
        self._load_cache()

    def _get_query_hash(self, query: str) -> str:
        """生成查询的哈希值"""
        return hashlib.sha256(content.encode()).hexdigest()

    def _save_cache(self):
        """保存缓存到磁盘"""
        cache_data = {
            'queries': self.queries,
            'responses': self.responses,
            'embeddings': self.embeddings,
            'dimension': self.dimension
        }

        # 保存元数据
        meta_path = self.cache_dir / "cache_meta.pkl"
        with open(meta_path, 'wb') as f:
            pickle.dump(cache_data, f)

        # 保存FAISS索引
        if self.index is not None:
            faiss.write_index(self.index, str(self.cache_dir / "faiss.index"))

    def _load_cache(self):
        """从磁盘加载缓存"""
        meta_path = self.cache_dir / "cache_meta.pkl"
        index_path = self.cache_dir / "faiss.index"

        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                cache_data = pickle.load(f)

            self.queries = cache_data['queries']
            self.responses = cache_data['responses']
            self.embeddings = cache_data['embeddings']
            self.dimension = cache_data['dimension']

            if index_path.exists() and self.dimension:
                self.index = faiss.read_index(str(index_path))

            print(f"加载语义缓存: {len(self.queries)} 条记录")

    def _init_index(self, dimension: int):
        """初始化FAISS索引"""
        self.dimension = dimension
        # 使用内积索引（适合余弦相似度）
        self.index = faiss.IndexFlatIP(dimension)

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """归一化向量（用于余弦相似度）"""
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def add(self, query: str, response: str):
        """添加缓存条目"""
        # 生成查询向量
        query_embedding = self.embedding_model.embed_query(query)
        query_vector = np.array(query_embedding).astype('float32').reshape(1, -1)

        # 归一化
        query_vector = self._normalize_vector(query_vector)

        # 初始化索引（如果是第一个）
        if self.index is None:
            self._init_index(query_vector.shape[1])

        # 添加到索引
        self.index.add(query_vector)

        # 存储数据
        self.queries.append(query)
        self.responses.append(response)
        self.embeddings.append(query_vector[0])

        # 保存到磁盘
        self._save_cache()

        print(f"添加缓存: {query[:50]}...")

    def get(self, query: str) -> Optional[str]:
        """获取缓存的响应"""
        if self.index is None or self.index.ntotal == 0:
            return None

        # 生成查询向量
        query_embedding = self.embedding_model.embed_query(query)
        query_vector = np.array(query_embedding).astype('float32').reshape(1, -1)
        query_vector = self._normalize_vector(query_vector)

        # 搜索最相似的
        scores, indices = self.index.search(query_vector, 1)

        # 检查相似度是否超过阈值
        if scores[0][0] > self.similarity_threshold:
            cached_response = self.responses[indices[0][0]]
            cached_query = self.queries[indices[0][0]]
            print(f"✓ 命中缓存 (相似度: {scores[0][0]:.3f}): {cached_query[:50]}...")
            return cached_response

        return None

    def clear(self):
        """清空缓存"""
        self.queries = []
        self.responses = []
        self.embeddings = []
        self.index = None
        self.dimension = None
        self._save_cache()
        print("语义缓存已清空")

    def stats(self) -> dict:
        """获取缓存统计信息"""
        return {
            "total_entries": len(self.queries),
            "dimension": self.dimension,
            "similarity_threshold": self.similarity_threshold
        }