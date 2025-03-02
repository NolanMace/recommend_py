import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import threading

from cache.lru_cache import LRUCache
from config.config_manager import get_config_manager

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class BaseCache:
    """缓存基类"""
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"cache.{name}")
    
    def get(self, key: str) -> Any:
        """获取缓存值"""
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """设置缓存值"""
        raise NotImplementedError
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        raise NotImplementedError
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        raise NotImplementedError
    
    def clear(self) -> bool:
        """清空缓存"""
        raise NotImplementedError

class MemoryCache(BaseCache):
    """本地内存缓存"""
    def __init__(self):
        super().__init__("memory")
        self._cache = {}  # 缓存数据
        self._expiry = {}  # 过期时间
    
    def get(self, key: str) -> Any:
        """获取缓存值"""
        if key not in self._cache:
            return None
        
        # 检查是否过期
        if key in self._expiry and self._expiry[key] < time.time():
            self.delete(key)
            return None
        
        return self._cache[key]
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间(秒)
        """
        self._cache[key] = value
        
        if ttl is not None:
            self._expiry[key] = time.time() + ttl
        elif key in self._expiry:
            # 如果不设置TTL但之前有设置，则删除过期时间
            del self._expiry[key]
            
        return True
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        if key in self._cache:
            del self._cache[key]
            if key in self._expiry:
                del self._expiry[key]
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        if key not in self._cache:
            return False
            
        # 检查是否过期
        if key in self._expiry and self._expiry[key] < time.time():
            self.delete(key)
            return False
            
        return True
    
    def clear(self) -> bool:
        """清空缓存"""
        self._cache.clear()
        self._expiry.clear()
        return True
    
    def cleanup(self) -> int:
        """清理过期数据，返回清理的数量"""
        count = 0
        now = time.time()
        expired_keys = [k for k, exp in self._expiry.items() if exp < now]
        
        for key in expired_keys:
            self.delete(key)
            count += 1
            
        return count

class RedisCache(BaseCache):
    """Redis缓存"""
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        super().__init__("redis")
        if not REDIS_AVAILABLE:
            raise ImportError("Redis模块未安装，请安装redis-py")
            
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True  # 自动解码为字符串
        )
        self.logger.info(f"Redis缓存已初始化: {host}:{port} db={db}")
    
    def get(self, key: str) -> Any:
        """获取缓存值"""
        value = self.client.get(key)
        if value is None:
            return None
            
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # 如果不是JSON格式，返回原始值
            return value
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """设置缓存值"""
        if not isinstance(value, (str, bytes)):
            value = json.dumps(value)
            
        if ttl is not None:
            return self.client.setex(key, ttl, value)
        else:
            return self.client.set(key, value)
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        return self.client.delete(key) > 0
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        return self.client.exists(key) > 0
    
    def clear(self) -> bool:
        """清空缓存"""
        self.client.flushdb()
        return True

class CacheManager:
    """缓存管理器
    
    提供统一的缓存接口，使用内存缓存（LRU策略）
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CacheManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """初始化缓存管理器"""
        if self._initialized:
            return
            
        self.logger = logging.getLogger("cache_manager")
        self.config = get_config_manager()
        
        # 缓存配置
        cache_config = self.config.get_section('cache')
        self.memory_cache_config = cache_config.get('memory', {})
        
        # 内存缓存启用状态
        self.memory_enabled = self.memory_cache_config.get('enabled', True)
        
        # 获取缓存过期时间策略
        self.cache_strategy = cache_config.get('strategy', {})
        self.hot_topics_ttl = self.cache_strategy.get('hot_topics_ttl', 300)
        self.user_recommendations_ttl = self.cache_strategy.get('user_recommendations_ttl', 3600)
        self.system_config_ttl = self.cache_strategy.get('system_config_ttl', 86400)
        
        # 创建缓存实例
        if self.memory_enabled:
            max_size = self.memory_cache_config.get('max_size', 10000)
            default_ttl = self.memory_cache_config.get('ttl', 3600)
            self.memory_cache = LRUCache(max_size=max_size, default_ttl=default_ttl)
            
            # 启动清理定时器
            self._start_cleanup_timer()
            
            self.logger.info(f"内存缓存已启用: max_size={max_size}, default_ttl={default_ttl}")
        else:
            self.memory_cache = None
            self.logger.warning("内存缓存已禁用")
        
        self._initialized = True
    
    def _start_cleanup_timer(self):
        """启动缓存清理定时器"""
        if not self.memory_enabled:
            return
            
        cleanup_interval = self.memory_cache_config.get('cleanup_interval', 300)
        
        def cleanup_task():
            """清理任务"""
            while True:
                time.sleep(cleanup_interval)
                try:
                    removed_count = self.memory_cache.cleanup()
                    if removed_count > 0:
                        self.logger.debug(f"内存缓存清理: 移除了 {removed_count} 个过期项")
                except Exception as e:
                    self.logger.error(f"内存缓存清理失败: {e}")
        
        # 创建并启动清理线程
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
        
        self.logger.debug(f"内存缓存清理定时器已启动, 间隔: {cleanup_interval}秒")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取缓存值
        
        Args:
            key: 缓存键
            default: 默认值，当缓存不存在时返回
            
        Returns:
            Any: 缓存值或默认值
        """
        if not self.memory_enabled:
            return default
            
        try:
            value = self.memory_cache.get(key)
            return value if value is not None else default
        except Exception as e:
            self.logger.error(f"获取缓存失败: {e}")
            return default
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间(秒)，None使用默认值
            
        Returns:
            bool: 是否设置成功
        """
        if not self.memory_enabled:
            return False
            
        try:
            return self.memory_cache.set(key, value, ttl)
        except Exception as e:
            self.logger.error(f"设置缓存失败: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """删除缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否删除成功
        """
        if not self.memory_enabled:
            return False
            
        try:
            return self.memory_cache.delete(key)
        except Exception as e:
            self.logger.error(f"删除缓存失败: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """检查缓存键是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否存在
        """
        if not self.memory_enabled:
            return False
            
        try:
            return self.memory_cache.exists(key)
        except Exception as e:
            self.logger.error(f"检查缓存是否存在失败: {e}")
            return False
    
    def clear(self) -> bool:
        """清空所有缓存
        
        Returns:
            bool: 是否清空成功
        """
        if not self.memory_enabled:
            return False
            
        try:
            return self.memory_cache.clear()
        except Exception as e:
            self.logger.error(f"清空缓存失败: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """获取缓存统计信息
        
        Returns:
            Dict: 统计信息字典
        """
        if not self.memory_enabled:
            return {"enabled": False}
        
        try:
            stats = self.memory_cache.get_stats()
            stats.update({"enabled": True})
            return stats
        except Exception as e:
            self.logger.error(f"获取缓存统计信息失败: {e}")
            return {"enabled": True, "error": str(e)}
    
    def get_hot_topic_key(self, topic_id: str) -> str:
        """获取热点话题缓存键
        
        Args:
            topic_id: 话题ID
            
        Returns:
            str: 缓存键
        """
        return f"hot_topic:{topic_id}"
    
    def set_hot_topic(self, topic_id: str, data: Dict) -> bool:
        """缓存热点话题
        
        Args:
            topic_id: 话题ID
            data: 话题数据
            
        Returns:
            bool: 是否设置成功
        """
        key = self.get_hot_topic_key(topic_id)
        return self.set(key, data, ttl=self.hot_topics_ttl)
    
    def get_hot_topic(self, topic_id: str) -> Optional[Dict]:
        """获取热点话题缓存
        
        Args:
            topic_id: 话题ID
            
        Returns:
            Optional[Dict]: 话题数据或None
        """
        key = self.get_hot_topic_key(topic_id)
        return self.get(key)
    
    def get_user_recommendation_key(self, user_id: str, scenario: str = 'default') -> str:
        """获取用户推荐结果缓存键
        
        Args:
            user_id: 用户ID
            scenario: 推荐场景
            
        Returns:
            str: 缓存键
        """
        return f"user_rec:{user_id}:{scenario}"
    
    def set_user_recommendations(self, user_id: str, recommendations: List, scenario: str = 'default') -> bool:
        """缓存用户推荐结果
        
        Args:
            user_id: 用户ID
            recommendations: 推荐结果列表
            scenario: 推荐场景
            
        Returns:
            bool: 是否设置成功
        """
        key = self.get_user_recommendation_key(user_id, scenario)
        return self.set(key, recommendations, ttl=self.user_recommendations_ttl)
    
    def get_user_recommendations(self, user_id: str, scenario: str = 'default') -> Optional[List]:
        """获取用户推荐结果缓存
        
        Args:
            user_id: 用户ID
            scenario: 推荐场景
            
        Returns:
            Optional[List]: 推荐结果列表或None
        """
        key = self.get_user_recommendation_key(user_id, scenario)
        return self.get(key)
    
    def get_config_key(self, config_name: str) -> str:
        """获取配置缓存键
        
        Args:
            config_name: 配置名称
            
        Returns:
            str: 缓存键
        """
        return f"config:{config_name}"
    
    def set_config(self, config_name: str, config_data: Any) -> bool:
        """缓存配置
        
        Args:
            config_name: 配置名称
            config_data: 配置数据
            
        Returns:
            bool: 是否设置成功
        """
        key = self.get_config_key(config_name)
        return self.set(key, config_data, ttl=self.system_config_ttl)
    
    def get_config(self, config_name: str) -> Optional[Any]:
        """获取配置缓存
        
        Args:
            config_name: 配置名称
            
        Returns:
            Optional[Any]: 配置数据或None
        """
        key = self.get_config_key(config_name)
        return self.get(key)

# 全局缓存管理器实例
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """获取缓存管理器实例
    
    Returns:
        CacheManager: 缓存管理器实例
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager 