# cache/cache_manager.py
from typing import Optional, Tuple
from app.cache.semantic_cache import SemanticCache
from app.cache.response_cache import MemoryCache, RedisCache


class CacheManager:
    """缓存管理器 - 整合多种缓存策略"""

    def __init__(self,
                 embedding_model,
                 semantic_threshold: float = 0.85,
                 use_redis: bool = False,
                 redis_url: str = "redis://localhost:6379"):
        """
        初始化缓存管理器

        Args:
            embedding_model: 嵌入模型
            semantic_threshold: 语义相似度阈值
            use_redis: 是否使用Redis
            redis_url: Redis连接URL
        """
        # 语义缓存（FAISS）
        self.semantic_cache = SemanticCache(
            embedding_model=embedding_model,
            similarity_threshold=semantic_threshold
        )

        # 响应缓存（精确匹配）
        if use_redis:
            self.response_cache = RedisCache(redis_url=redis_url)
        else:
            self.response_cache = MemoryCache(max_size=1000)

    def get(self, query: str, context_hash: str = "") -> Tuple[Optional[str], str]:
        """
        获取缓存（多级缓存策略）

        Returns:
            (response, cache_level): 响应和缓存级别
        """
        # 第一级：精确匹配缓存
        response = self.response_cache.get(query, context_hash)
        if response:
            return response, "exact_match"

        # 第二级：语义缓存
        response = self.semantic_cache.get(query)
        if response:
            return response, "semantic_match"

        return None, "miss"

    def set(self, query: str, response: str, context_hash: str = ""):
        """设置缓存"""
        # 同时添加到两种缓存
        self.response_cache.set(query, response, context_hash)
        self.semantic_cache.add(query, response)

    def clear(self):
        """清空所有缓存"""
        self.semantic_cache.clear()
        self.response_cache.clear()
        print("所有缓存已清空")

    def stats(self) -> dict:
        """获取缓存统计"""
        return {
            "semantic_cache": self.semantic_cache.stats(),
            "response_cache": self.response_cache.stats()
        }