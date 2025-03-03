# -*- coding: utf-8 -*-
"""
缓存管理模块：提供内存缓存和持久化缓存功能
"""
import time
import json
import pickle
import threading
import os
from datetime import datetime, timedelta
from functools import wraps

# 简单的内存缓存实现
class MemoryCache:
    def __init__(self, max_size=1000, expire_seconds=3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.expire_seconds = expire_seconds
        self.lock = threading.RLock()
    
    def get(self, key):
        """获取缓存值，如果过期或不存在返回None"""
        with self.lock:
            if key not in self.cache:
                return None
            
            value, timestamp = self.cache[key]
            current_time = time.time()
            
            # 检查是否过期
            if current_time - timestamp > self.expire_seconds:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                return None
            
            # 更新访问时间
            self.access_times[key] = current_time
            return value
    
    def set(self, key, value, expire_seconds=None):
        """设置缓存值"""
        with self.lock:
            # 检查缓存大小，如果达到上限则清理最少使用的
            if len(self.cache) >= self.max_size:
                self._cleanup()
            
            # 设置缓存
            timestamp = time.time()
            self.cache[key] = (value, timestamp)
            self.access_times[key] = timestamp
            
            # 更新过期时间
            if expire_seconds:
                self.expire_seconds = expire_seconds
    
    def delete(self, key):
        """删除指定键的缓存"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
    
    def clear(self):
        """清空所有缓存"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def _cleanup(self):
        """清理最少使用的缓存项"""
        if not self.access_times:
            return
            
        # 按访问时间排序找出最旧的20%缓存项删除
        sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
        remove_count = max(1, int(len(sorted_keys) * 0.2))
        
        for i in range(remove_count):
            key = sorted_keys[i][0]
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]

# 持久化磁盘缓存
class DiskCache:
    def __init__(self, cache_dir='.cache', expire_seconds=86400):
        self.cache_dir = cache_dir
        self.expire_seconds = expire_seconds
        self.lock = threading.RLock()
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key):
        """获取缓存文件路径"""
        # 替换非法文件名字符
        safe_key = str(key).replace('/', '_').replace('\\', '_').replace(':', '_')
        return os.path.join(self.cache_dir, f"{safe_key}.cache")
    
    def get(self, key):
        """从磁盘获取缓存值"""
        cache_path = self._get_cache_path(key)
        
        with self.lock:
            if not os.path.exists(cache_path):
                return None
            
            # 检查文件是否过期
            file_time = os.path.getmtime(cache_path)
            if time.time() - file_time > self.expire_seconds:
                try:
                    os.remove(cache_path)
                except:
                    pass
                return None
            
            # 读取缓存
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
    
    def set(self, key, value, expire_seconds=None):
        """将值保存到磁盘缓存"""
        cache_path = self._get_cache_path(key)
        
        with self.lock:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
            except Exception as e:
                print(f"保存缓存失败: {str(e)}")
    
    def delete(self, key):
        """删除缓存文件"""
        cache_path = self._get_cache_path(key)
        
        with self.lock:
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except:
                    pass
    
    def clear(self):
        """清空所有缓存文件"""
        with self.lock:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    try:
                        os.remove(os.path.join(self.cache_dir, filename))
                    except:
                        pass

# 缓存装饰器
def cache_result(cache, key_prefix='', expire_seconds=None):
    """缓存函数结果的装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            key_parts = [key_prefix, func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            cache_key = "_".join(key_parts)
            
            # 尝试从缓存获取
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数获取结果
            result = func(*args, **kwargs)
            
            # 存入缓存
            cache.set(cache_key, result, expire_seconds)
            return result
        return wrapper
    return decorator

# 全局缓存实例
memory_cache = MemoryCache(max_size=5000, expire_seconds=1800)  # 30分钟过期
disk_cache = DiskCache(cache_dir='.recommend_cache', expire_seconds=86400)  # 1天过期

# 缓存统计
cache_stats = {
    'hits': 0,
    'misses': 0,
    'total': 0
}
cache_stats_lock = threading.Lock()

def get_cache_stats():
    """获取缓存命中统计"""
    with cache_stats_lock:
        stats = cache_stats.copy()
        hit_ratio = stats['hits'] / stats['total'] if stats['total'] > 0 else 0
        stats['hit_ratio'] = f"{hit_ratio:.2%}"
        return stats

def reset_cache_stats():
    """重置缓存统计"""
    with cache_stats_lock:
        for key in cache_stats:
            cache_stats[key] = 0

# 带统计的缓存获取
def cached_get(cache, key, loader_func=None, expire_seconds=None):
    """获取缓存，如果不存在则调用loader_func加载"""
    with cache_stats_lock:
        cache_stats['total'] += 1
    
    # 尝试从缓存获取
    result = cache.get(key)
    
    if result is not None:
        with cache_stats_lock:
            cache_stats['hits'] += 1
        return result
    
    # 缓存未命中，调用加载函数
    with cache_stats_lock:
        cache_stats['misses'] += 1
    
    if loader_func:
        result = loader_func()
        if result is not None:
            cache.set(key, result, expire_seconds)
        return result
    
    return None

# 计时器上下文管理器
class TimerContext:
    def __init__(self, name=None, verbose=True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        if self.verbose and self.name:
            print(f"{self.name} 执行时间: {self.elapsed:.4f}秒")
        return False  # 不抑制异常 