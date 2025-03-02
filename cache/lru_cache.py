import time
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict

class LRUCache:
    """LRU (Least Recently Used) 缓存实现
    
    基于OrderedDict实现的LRU缓存，支持最大容量限制和过期时间
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = None):
        """初始化LRU缓存
        
        Args:
            max_size: 最大缓存条目数
            default_ttl: 默认过期时间(秒)，None表示不过期
        """
        self.logger = logging.getLogger("lru_cache")
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()  # 缓存数据: {key: (value, expire_time)}
        self.lock = threading.RLock()  # 可重入锁，支持在同一线程中多次获取
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        
    def get(self, key: str) -> Any:
        """获取缓存值
        
        如果键存在且未过期，将其移到OrderedDict的末尾（最近使用）
        
        Args:
            key: 缓存键
            
        Returns:
            Any: 缓存值，不存在或已过期返回None
        """
        with self.lock:
            if key not in self.cache:
                self.miss_count += 1
                return None
                
            value, expire_time = self.cache[key]
            
            # 检查是否过期
            if expire_time is not None and expire_time < time.time():
                self.cache.pop(key)
                self.miss_count += 1
                return None
                
            # 移动到OrderedDict末尾，表示最近使用
            self.cache.move_to_end(key)
            self.hit_count += 1
            return value
            
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间(秒)，None使用默认值，-1表示永不过期
            
        Returns:
            bool: 是否设置成功
        """
        with self.lock:
            # 计算过期时间
            expire_time = None
            if ttl is None:
                if self.default_ttl is not None:
                    expire_time = time.time() + self.default_ttl
            elif ttl > 0:
                expire_time = time.time() + ttl
                
            # 如果键已存在，更新并移到末尾
            if key in self.cache:
                self.cache[key] = (value, expire_time)
                self.cache.move_to_end(key)
                return True
                
            # 如果达到最大容量，移除最久未使用的项
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)  # last=False表示移除第一个项
                self.eviction_count += 1
                
            # 添加新项到末尾
            self.cache[key] = (value, expire_time)
            return True
            
    def delete(self, key: str) -> bool:
        """删除缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否删除成功
        """
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
                return True
            return False
            
    def exists(self, key: str) -> bool:
        """检查键是否存在且未过期
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 键是否存在且未过期
        """
        with self.lock:
            if key not in self.cache:
                return False
                
            _, expire_time = self.cache[key]
            
            # 检查是否过期
            if expire_time is not None and expire_time < time.time():
                self.cache.pop(key)
                return False
                
            return True
            
    def clear(self) -> bool:
        """清空缓存
        
        Returns:
            bool: 是否清空成功
        """
        with self.lock:
            self.cache.clear()
            return True
            
    def cleanup(self) -> int:
        """清理过期数据
        
        Returns:
            int: 清理的数量
        """
        count = 0
        now = time.time()
        
        with self.lock:
            # 找出所有过期的键
            expired_keys = [
                key for key, (_, expire_time) in self.cache.items()
                if expire_time is not None and expire_time < now
            ]
            
            # 删除过期的键
            for key in expired_keys:
                self.cache.pop(key)
                count += 1
                
        return count
        
    def get_stats(self) -> Dict:
        """获取缓存统计信息
        
        Returns:
            Dict: 统计信息字典
        """
        with self.lock:
            total = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total if total > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'eviction_count': self.eviction_count
            }
            
    def get_keys(self) -> List[str]:
        """获取所有缓存键
        
        Returns:
            List[str]: 缓存键列表
        """
        with self.lock:
            return list(self.cache.keys())
            
    def get_oldest(self, count: int = 1) -> List[Tuple[str, Any]]:
        """获取最久未使用的缓存项
        
        Args:
            count: 获取数量
            
        Returns:
            List[Tuple[str, Any]]: (键, 值)元组列表
        """
        with self.lock:
            result = []
            for key in list(self.cache.keys())[:count]:
                value, _ = self.cache[key]
                result.append((key, value))
            return result
            
    def get_newest(self, count: int = 1) -> List[Tuple[str, Any]]:
        """获取最近使用的缓存项
        
        Args:
            count: 获取数量
            
        Returns:
            List[Tuple[str, Any]]: (键, 值)元组列表
        """
        with self.lock:
            result = []
            for key in reversed(list(self.cache.keys())[-count:]):
                value, _ = self.cache[key]
                result.append((key, value))
            return result
            
    def __len__(self) -> int:
        """获取缓存大小
        
        Returns:
            int: 缓存条目数
        """
        return len(self.cache) 