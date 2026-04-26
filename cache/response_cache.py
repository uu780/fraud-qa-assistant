# cache/response_cache.py
from typing import Optional, Dict
from collections import OrderedDict
import json
import hashlib


class MemoryCache:
    """内存缓存（LRU策略）"""

    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size

    def _get_key(self, query: str, context_hash: str = "") -> str:
        """生成缓存键"""
        content = f"{query}|{context_hash}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, query: str, context_hash: str = "") -> Optional[str]:
        """获取缓存"""
        key = self._get_key(query, context_hash)
        if key in self.cache:
            # 移动到末尾（最近使用）
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, query: str, response: str, context_hash: str = ""):
        """设置缓存"""
        key = self._get_key(query, context_hash)

        # 如果已存在，先删除
        if key in self.cache:
            del self.cache[key]

        # 如果超过最大容量，删除最旧的
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        # 添加新缓存
        self.cache[key] = response

    def clear(self):
        """清空缓存"""
        self.cache.clear()

    def stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_entries": len(self.cache),
            "max_size": self.max_size
        }


class RedisCache:
    """Redis缓存（需要安装redis）"""

    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        try:
            import redis
            self.redis_client = redis.from_url(redis_url)
            self.ttl = ttl
            self.enabled = True
        except ImportError:
            print("Redis未安装，使用内存缓存")
            self.enabled = False
        except Exception as e:
            print(f"Redis连接失败: {e}")
            self.enabled = False

    def _get_key(self, query: str, context_hash: str = "") -> str:
        """生成缓存键"""
        content = f"rag:response:{query}:{context_hash}"
        return hashlib.sha256(content.encode()).hexdigest() 

    def get(self, query: str, context_hash: str = "") -> Optional[str]:
        """获取缓存"""
        if not self.enabled:
            return None

        key = self._get_key(query, context_hash)
        value = self.redis_client.get(key)
        return value.decode() if value else None

    def set(self, query: str, response: str, context_hash: str = ""):
        """设置缓存"""
        if not self.enabled:
            return

        key = self._get_key(query, context_hash)
        self.redis_client.setex(key, self.ttl, response)

    def clear(self):
        """清空缓存"""
        if not self.enabled:
            return

        pattern = "rag:response:*"
        keys = self.redis_client.keys(pattern)
        if keys:
            self.redis_client.delete(*keys)