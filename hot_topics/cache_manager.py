from typing import Dict, List, Optional
from datetime import datetime
from cache import memory_cache

class HotTopicsCacheManager:
    """热点话题缓存管理器"""
    
    HOT_TOPICS_CACHE_KEY = "hot_topics"
    
    def __init__(self):
        self.cache = memory_cache
    
    def get_hot_topics(self) -> Optional[Dict]:
        """获取缓存的热点话题数据"""
        return self.cache.get(self.HOT_TOPICS_CACHE_KEY)
    
    def set_hot_topics(self, hot_topics: List[Dict], ttl: int = 300) -> None:
        """设置热点话题缓存
        
        Args:
            hot_topics: 热点话题列表
            ttl: 缓存过期时间（秒）
        """
        cache_data = {
            'hot_topics': hot_topics,
            'updated_at': datetime.now().isoformat()
        }
        self.cache.set(self.HOT_TOPICS_CACHE_KEY, cache_data, expire_seconds=ttl)
    
    def clear_hot_topics(self) -> None:
        """清除热点话题缓存"""
        self.cache.delete(self.HOT_TOPICS_CACHE_KEY) 