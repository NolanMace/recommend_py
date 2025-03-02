import logging
import time
import random
import threading
from typing import List, Dict, Any, Set, Optional
from datetime import datetime, timedelta

class ExposurePoolManager:
    """曝光池管理器
    
    管理多个曝光池，包括新内容池、热门内容池、优质内容池和多样性内容池。
    支持从不同池中按权重混合获取内容，用于推荐系统的兜底策略和无限滚动场景。
    """
    
    def __init__(self, db_pool, cache_manager=None):
        """初始化曝光池管理器
        
        Args:
            db_pool: 数据库连接池
            cache_manager: 缓存管理器，可选
        """
        self.db_pool = db_pool
        self.cache_manager = cache_manager
        self.logger = logging.getLogger("exposure_pool")
        
        # 各曝光池容量
        self.pool_capacity = {
            'new': 500,      # 新内容池容量
            'hot': 300,      # 热门内容池容量
            'quality': 200,  # 优质内容池容量
            'diverse': 100   # 多样性内容池容量
        }
        
        # 各曝光池默认权重
        self.default_weights = {
            'new': 0.3,      # 新内容权重
            'hot': 0.4,      # 热门内容权重
            'quality': 0.2,  # 优质内容权重
            'diverse': 0.1   # 多样性内容权重
        }
        
        # 曝光池缓存键
        self.pool_cache_keys = {
            'new': 'exposure_pool:new',
            'hot': 'exposure_pool:hot',
            'quality': 'exposure_pool:quality',
            'diverse': 'exposure_pool:diverse'
        }
        
        # 曝光池刷新间隔（秒）
        self.refresh_intervals = {
            'new': 5 * 60,       # 5分钟
            'hot': 15 * 60,      # 15分钟
            'quality': 60 * 60,  # 1小时
            'diverse': 3 * 60 * 60  # 3小时
        }
        
        # 曝光池内容
        self.pools = {
            'new': [],
            'hot': [],
            'quality': [],
            'diverse': []
        }
        
        # 曝光池最后更新时间
        self.last_refresh = {
            'new': datetime.min,
            'hot': datetime.min,
            'quality': datetime.min,
            'diverse': datetime.min
        }
        
        # 初始化锁
        self.locks = {
            'new': threading.RLock(),
            'hot': threading.RLock(),
            'quality': threading.RLock(),
            'diverse': threading.RLock()
        }
        
        # 初始化曝光池
        self.init_pools()
        
    def init_pools(self):
        """初始化所有曝光池"""
        for pool_type in self.pools.keys():
            self.refresh_pool(pool_type)
            
    def refresh_pool(self, pool_type: str) -> bool:
        """刷新指定曝光池
        
        Args:
            pool_type: 曝光池类型，可选值: new, hot, quality, diverse
            
        Returns:
            刷新是否成功
        """
        if pool_type not in self.pools:
            self.logger.error(f"未知的曝光池类型: {pool_type}")
            return False
            
        # 检查是否需要刷新
        now = datetime.now()
        if (now - self.last_refresh[pool_type]).total_seconds() < self.refresh_intervals[pool_type]:
            return True  # 不需要刷新
            
        # 尝试从缓存获取
        if self.cache_manager:
            cache_key = self.pool_cache_keys[pool_type]
            cached_pool = self.cache_manager.get(cache_key)
            if cached_pool:
                with self.locks[pool_type]:
                    self.pools[pool_type] = cached_pool
                    self.last_refresh[pool_type] = now
                self.logger.info(f"从缓存刷新 {pool_type} 曝光池，获取 {len(cached_pool)} 条内容")
                return True
                
        # 从数据库刷新
        try:
            conn = self.db_pool.get_connection()
            try:
                cursor = conn.cursor()
                
                # 根据不同池类型执行不同查询
                if pool_type == 'new':
                    # 新内容池：最近发布的内容
                    cursor.execute("""
                        SELECT p.post_id, p.title, p.author_id, p.created_at, p.category_id, 
                               p.view_count, p.like_count, p.comment_count
                        FROM posts p
                        WHERE p.status = 'published' 
                        AND p.created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
                        ORDER BY p.created_at DESC
                        LIMIT %s
                    """, (self.pool_capacity[pool_type],))
                    
                elif pool_type == 'hot':
                    # 热门内容池：最近互动最多的内容
                    cursor.execute("""
                        SELECT p.post_id, p.title, p.author_id, p.created_at, p.category_id,
                               p.view_count, p.like_count, p.comment_count,
                               (p.view_count * 0.1 + p.like_count * 0.5 + p.comment_count * 0.4) as hot_score
                        FROM posts p
                        WHERE p.status = 'published'
                        AND p.created_at > DATE_SUB(NOW(), INTERVAL 30 DAY)
                        ORDER BY hot_score DESC
                        LIMIT %s
                    """, (self.pool_capacity[pool_type],))
                    
                elif pool_type == 'quality':
                    # 优质内容池：评分最高的内容
                    cursor.execute("""
                        SELECT p.post_id, p.title, p.author_id, p.created_at, p.category_id,
                               p.view_count, p.like_count, p.comment_count, p.rating
                        FROM posts p
                        WHERE p.status = 'published'
                        AND p.rating >= 4.0
                        ORDER BY p.rating DESC, p.created_at DESC
                        LIMIT %s
                    """, (self.pool_capacity[pool_type],))
                    
                elif pool_type == 'diverse':
                    # 多样性内容池：不同类别的优质内容
                    cursor.execute("""
                        SELECT p.post_id, p.title, p.author_id, p.created_at, p.category_id,
                               p.view_count, p.like_count, p.comment_count, c.category_name
                        FROM (
                            SELECT category_id, 
                                   @row_number:=CASE
                                       WHEN @current_category = category_id THEN @row_number + 1
                                       ELSE 1
                                   END AS row_number,
                                   @current_category:=category_id
                            FROM posts
                            JOIN (SELECT @row_number:=0, @current_category:=0) as vars
                            WHERE status = 'published'
                            ORDER BY category_id, 
                                     (view_count * 0.2 + like_count * 0.4 + comment_count * 0.4) DESC
                        ) as ranked_posts
                        JOIN posts p ON p.category_id = ranked_posts.category_id
                        JOIN categories c ON c.category_id = p.category_id
                        WHERE ranked_posts.row_number <= 10
                        ORDER BY p.category_id, p.created_at DESC
                        LIMIT %s
                    """, (self.pool_capacity[pool_type],))
                
                # 获取结果
                rows = cursor.fetchall()
                cursor.close()
                
                # 转换为字典列表
                pool_items = []
                for row in rows:
                    item = {
                        'post_id': row[0],
                        'title': row[1],
                        'author_id': row[2],
                        'created_at': row[3].isoformat() if hasattr(row[3], 'isoformat') else row[3],
                        'category_id': row[4],
                        'view_count': row[5],
                        'like_count': row[6],
                        'comment_count': row[7],
                        'pool_type': pool_type
                    }
                    
                    # 添加特定字段
                    if pool_type == 'hot' and len(row) > 8:
                        item['hot_score'] = float(row[8])
                    elif pool_type == 'quality' and len(row) > 8:
                        item['rating'] = float(row[8])
                    elif pool_type == 'diverse' and len(row) > 8:
                        item['category_name'] = row[8]
                        
                    pool_items.append(item)
                
                # 更新曝光池
                with self.locks[pool_type]:
                    self.pools[pool_type] = pool_items
                    self.last_refresh[pool_type] = now
                
                # 更新缓存
                if self.cache_manager:
                    # 设置缓存，过期时间为刷新间隔的2倍
                    ttl = self.refresh_intervals[pool_type] * 2
                    self.cache_manager.set(self.pool_cache_keys[pool_type], pool_items, ttl=ttl)
                
                self.logger.info(f"从数据库刷新 {pool_type} 曝光池，获取 {len(pool_items)} 条内容")
                return True
                
            finally:
                self.db_pool.release_connection(conn)
                
        except Exception as e:
            self.logger.error(f"刷新 {pool_type} 曝光池时出错: {e}")
            return False
    
    def get_pool_items(self, pool_type: str, count: int, excluded_ids: Optional[Set] = None) -> List[Dict]:
        """从指定曝光池获取内容
        
        Args:
            pool_type: 曝光池类型
            count: 获取数量
            excluded_ids: 需要排除的内容ID集合
            
        Returns:
            内容列表
        """
        if pool_type not in self.pools:
            self.logger.error(f"未知的曝光池类型: {pool_type}")
            return []
            
        # 检查是否需要刷新
        now = datetime.now()
        if (now - self.last_refresh[pool_type]).total_seconds() > self.refresh_intervals[pool_type]:
            self.refresh_pool(pool_type)
            
        # 获取曝光池内容
        with self.locks[pool_type]:
            pool_items = self.pools[pool_type].copy()
            
        # 排除指定ID
        if excluded_ids:
            pool_items = [item for item in pool_items if item['post_id'] not in excluded_ids]
            
        # 如果数量不足，尝试刷新
        if len(pool_items) < count:
            self.refresh_pool(pool_type)
            with self.locks[pool_type]:
                pool_items = self.pools[pool_type].copy()
            if excluded_ids:
                pool_items = [item for item in pool_items if item['post_id'] not in excluded_ids]
                
        # 随机打乱顺序，增加多样性
        random.shuffle(pool_items)
        
        # 返回指定数量
        return pool_items[:count]
    
    def get_mixed_items(self, count: int, excluded_ids: Optional[Set] = None, 
                       pool_weights: Optional[Dict[str, float]] = None) -> List[Dict]:
        """从多个曝光池按权重混合获取内容
        
        Args:
            count: 获取总数量
            excluded_ids: 需要排除的内容ID集合
            pool_weights: 各曝光池权重，不指定则使用默认权重
            
        Returns:
            混合内容列表
        """
        if count <= 0:
            return []
            
        # 使用指定权重或默认权重
        weights = pool_weights or self.default_weights
        
        # 确保权重有效
        valid_pools = [p for p in weights.keys() if p in self.pools]
        if not valid_pools:
            self.logger.error("没有有效的曝光池权重")
            return []
            
        # 归一化权重
        total_weight = sum(weights[p] for p in valid_pools)
        if total_weight <= 0:
            self.logger.error("曝光池总权重必须大于0")
            return []
            
        normalized_weights = {p: weights[p] / total_weight for p in valid_pools}
        
        # 计算各池应获取的数量
        pool_counts = {}
        remaining = count
        for pool in valid_pools[:-1]:  # 除了最后一个池
            pool_count = int(count * normalized_weights[pool])
            pool_counts[pool] = pool_count
            remaining -= pool_count
        
        # 最后一个池获取剩余数量
        pool_counts[valid_pools[-1]] = remaining
        
        # 从各池获取内容
        mixed_items = []
        used_ids = set(excluded_ids) if excluded_ids else set()
        
        for pool_type, pool_count in pool_counts.items():
            if pool_count <= 0:
                continue
                
            # 获取内容
            items = self.get_pool_items(pool_type, pool_count, used_ids)
            
            # 更新已使用ID
            for item in items:
                used_ids.add(item['post_id'])
                
            mixed_items.extend(items)
            
        # 如果获取的内容不足，尝试从其他池补充
        if len(mixed_items) < count:
            self.logger.info(f"混合获取内容数量不足 ({len(mixed_items)}/{count})，尝试补充")
            
            # 按权重排序池
            sorted_pools = sorted(valid_pools, key=lambda p: normalized_weights[p], reverse=True)
            
            # 依次从各池补充
            for pool_type in sorted_pools:
                if len(mixed_items) >= count:
                    break
                    
                # 计算需要补充的数量
                needed = count - len(mixed_items)
                
                # 获取补充内容
                additional_items = self.get_pool_items(pool_type, needed, used_ids)
                
                # 更新已使用ID
                for item in additional_items:
                    used_ids.add(item['post_id'])
                    
                mixed_items.extend(additional_items)
                
        # 随机打乱顺序
        random.shuffle(mixed_items)
        
        return mixed_items[:count]
    
    def refresh_all_pools(self) -> Dict[str, bool]:
        """刷新所有曝光池
        
        Returns:
            各池刷新结果
        """
        results = {}
        for pool_type in self.pools.keys():
            results[pool_type] = self.refresh_pool(pool_type)
        return results
    
    def get_pool_stats(self) -> Dict[str, Dict]:
        """获取各曝光池统计信息
        
        Returns:
            统计信息字典
        """
        stats = {}
        for pool_type in self.pools.keys():
            with self.locks[pool_type]:
                stats[pool_type] = {
                    'size': len(self.pools[pool_type]),
                    'capacity': self.pool_capacity[pool_type],
                    'last_refresh': self.last_refresh[pool_type].isoformat(),
                    'next_refresh': (self.last_refresh[pool_type] + 
                                    timedelta(seconds=self.refresh_intervals[pool_type])).isoformat()
                }
        return stats 