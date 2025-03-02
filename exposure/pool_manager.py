import logging
import random
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class ExposurePool:
    """曝光池基类"""
    def __init__(self, name: str, capacity: int, weight: float = 1.0):
        self.name = name
        self.capacity = capacity
        self.weight = weight  # 在混合推荐中的权重
        self.posts = []
        self.logger = logging.getLogger(f"exposure.pool.{name}")
        
    def add_post(self, post: Dict[str, Any]) -> bool:
        """添加帖子到曝光池"""
        if len(self.posts) >= self.capacity:
            return False
            
        self.posts.append(post)
        return True
        
    def remove_post(self, post_id: int) -> bool:
        """从曝光池中移除帖子"""
        for i, post in enumerate(self.posts):
            if post['post_id'] == post_id:
                del self.posts[i]
                return True
        return False
        
    def get_posts(self, count: int = 10, exclude_ids: List[int] = None) -> List[Dict]:
        """获取指定数量的帖子
        
        Args:
            count: 获取数量
            exclude_ids: 排除的帖子ID列表
        """
        exclude_ids = exclude_ids or []
        candidate_posts = [p for p in self.posts if p['post_id'] not in exclude_ids]
        
        if not candidate_posts:
            return []
            
        # 默认实现：随机选择
        selected = random.sample(
            candidate_posts, 
            min(count, len(candidate_posts))
        )
        
        # 添加来源标记
        for post in selected:
            post['pool'] = self.name
            
        return selected
        
    def refresh(self, posts_df: pd.DataFrame) -> int:
        """刷新曝光池内容"""
        raise NotImplementedError
        
    def __len__(self) -> int:
        return len(self.posts)

class NewPostsPool(ExposurePool):
    """新帖曝光池"""
    def __init__(self, capacity: int = 1000, weight: float = 0.5, max_age_hours: int = 24):
        super().__init__("new_posts", capacity, weight)
        self.max_age_hours = max_age_hours
        
    def refresh(self, posts_df: pd.DataFrame) -> int:
        """刷新新帖曝光池
        
        从最近max_age_hours小时内发布的帖子中选择，按发布时间倒序
        """
        # 清空当前池
        self.posts = []
        
        # 计算时间阈值
        cutoff_time = datetime.now() - timedelta(hours=self.max_age_hours)
        
        # 过滤最近发布的帖子
        recent_posts = posts_df[
            pd.to_datetime(posts_df['created_at']) > cutoff_time
        ]
        
        # 按发布时间倒序排序
        recent_posts = recent_posts.sort_values('created_at', ascending=False)
        
        # 填充曝光池
        count = min(self.capacity, len(recent_posts))
        self.posts = recent_posts.head(count).to_dict('records')
        
        self.logger.info(f"新帖曝光池已刷新，共{len(self.posts)}条")
        return len(self.posts)
        
    def get_posts(self, count: int = 10, exclude_ids: List[int] = None) -> List[Dict]:
        """获取帖子，优先返回最新的"""
        exclude_ids = exclude_ids or []
        candidate_posts = [p for p in self.posts if p['post_id'] not in exclude_ids]
        
        if not candidate_posts:
            return []
            
        # 按发布时间降序排序
        sorted_posts = sorted(
            candidate_posts, 
            key=lambda x: x.get('created_at', ''), 
            reverse=True
        )
        
        selected = sorted_posts[:min(count, len(sorted_posts))]
        
        # 添加来源标记
        for post in selected:
            post['pool'] = self.name
            
        return selected

class HotPostsPool(ExposurePool):
    """热门帖子曝光池"""
    def __init__(self, capacity: int = 500, weight: float = 0.3, min_heat_score: float = 10.0):
        super().__init__("hot_posts", capacity, weight)
        self.min_heat_score = min_heat_score
        
    def refresh(self, posts_df: pd.DataFrame) -> int:
        """刷新热门帖子曝光池
        
        选择热度分数超过阈值的帖子，按热度降序
        """
        # 清空当前池
        self.posts = []
        
        # 过滤高热度帖子
        hot_posts = posts_df[posts_df['heat_score'] >= self.min_heat_score]
        
        # 按热度降序排序
        hot_posts = hot_posts.sort_values('heat_score', ascending=False)
        
        # 填充曝光池
        count = min(self.capacity, len(hot_posts))
        self.posts = hot_posts.head(count).to_dict('records')
        
        self.logger.info(f"热门帖子曝光池已刷新，共{len(self.posts)}条")
        return len(self.posts)
        
    def get_posts(self, count: int = 10, exclude_ids: List[int] = None) -> List[Dict]:
        """获取帖子，优先返回热度高的"""
        exclude_ids = exclude_ids or []
        candidate_posts = [p for p in self.posts if p['post_id'] not in exclude_ids]
        
        if not candidate_posts:
            return []
            
        # 按热度降序排序
        sorted_posts = sorted(
            candidate_posts, 
            key=lambda x: x.get('heat_score', 0), 
            reverse=True
        )
        
        selected = sorted_posts[:min(count, len(sorted_posts))]
        
        # 添加来源标记
        for post in selected:
            post['pool'] = self.name
            
        return selected

class QualityPostsPool(ExposurePool):
    """优质帖子曝光池，对于热度中等但是质量高的帖子"""
    def __init__(self, capacity: int = 300, weight: float = 0.2, 
                 min_quality_score: float = 0.7, max_age_days: int = 30):
        super().__init__("quality_posts", capacity, weight)
        self.min_quality_score = min_quality_score
        self.max_age_days = max_age_days
        
    def refresh(self, posts_df: pd.DataFrame) -> int:
        """刷新优质帖子曝光池
        
        选择质量分数高但热度中等的帖子，并且时间不超过max_age_days
        """
        # 清空当前池
        self.posts = []
        
        # 计算时间阈值
        cutoff_time = datetime.now() - timedelta(days=self.max_age_days)
        
        # 过滤符合条件的帖子
        quality_posts = posts_df[
            (pd.to_datetime(posts_df['created_at']) > cutoff_time) &
            (posts_df['quality_score'] >= self.min_quality_score)
        ]
        
        # 按质量分数降序排序
        quality_posts = quality_posts.sort_values('quality_score', ascending=False)
        
        # 填充曝光池
        count = min(self.capacity, len(quality_posts))
        self.posts = quality_posts.head(count).to_dict('records')
        
        self.logger.info(f"优质帖子曝光池已刷新，共{len(self.posts)}条")
        return len(self.posts)
        
    def get_posts(self, count: int = 10, exclude_ids: List[int] = None) -> List[Dict]:
        """获取帖子，优先返回质量高的"""
        exclude_ids = exclude_ids or []
        candidate_posts = [p for p in self.posts if p['post_id'] not in exclude_ids]
        
        if not candidate_posts:
            return []
            
        # 按质量分数降序排序
        sorted_posts = sorted(
            candidate_posts, 
            key=lambda x: x.get('quality_score', 0), 
            reverse=True
        )
        
        selected = sorted_posts[:min(count, len(sorted_posts))]
        
        # 添加来源标记
        for post in selected:
            post['pool'] = self.name
            
        return selected

class DiversityPostsPool(ExposurePool):
    """多样性帖子曝光池，保证内容多样性"""
    def __init__(self, capacity: int = 200, weight: float = 0.1, 
                 categories: List[str] = None, category_field: str = 'category'):
        super().__init__("diversity_posts", capacity, weight)
        self.categories = categories or []
        self.category_field = category_field
        self.category_quotas = {}  # 每个类别的配额
        
    def refresh(self, posts_df: pd.DataFrame) -> int:
        """刷新多样性帖子曝光池
        
        为每个类别分配配额，确保多样性
        """
        # 清空当前池
        self.posts = []
        
        # 如果没有预设类别，从数据中获取
        if not self.categories:
            self.categories = posts_df[self.category_field].unique().tolist()
            
        # 计算每个类别的配额
        total_categories = len(self.categories)
        if total_categories > 0:
            base_quota = max(1, self.capacity // total_categories)
            
            self.category_quotas = {
                category: base_quota for category in self.categories
            }
            
            # 处理余数
            remainder = self.capacity - (base_quota * total_categories)
            for i, category in enumerate(self.categories):
                if i < remainder:
                    self.category_quotas[category] += 1
        
        # 按类别填充池
        for category, quota in self.category_quotas.items():
            # 过滤该类别的帖子
            category_posts = posts_df[posts_df[self.category_field] == category]
            
            # 随机选择该类别的帖子
            if len(category_posts) > 0:
                count = min(quota, len(category_posts))
                sampled = category_posts.sample(n=count)
                self.posts.extend(sampled.to_dict('records'))
        
        self.logger.info(f"多样性帖子曝光池已刷新，共{len(self.posts)}条")
        return len(self.posts)
        
    def get_posts(self, count: int = 10, exclude_ids: List[int] = None) -> List[Dict]:
        """获取帖子，确保类别多样性"""
        exclude_ids = exclude_ids or []
        candidate_posts = [p for p in self.posts if p['post_id'] not in exclude_ids]
        
        if not candidate_posts:
            return []
            
        # 按类别分组
        category_groups = {}
        for post in candidate_posts:
            category = post.get(self.category_field, 'unknown')
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(post)
            
        # 确保多样性选择
        selected = []
        categories = list(category_groups.keys())
        
        # 计算每个类别应选的数量
        if categories:
            base_count = max(1, count // len(categories))
            
            # 轮流从每个类别中选择
            for category in categories:
                group = category_groups[category]
                group_sample = random.sample(
                    group, 
                    min(base_count, len(group))
                )
                selected.extend(group_sample)
                
            # 如果还需要更多帖子，随机选择
            remaining = count - len(selected)
            if remaining > 0:
                # 剩余可选的帖子
                remaining_posts = [
                    p for p in candidate_posts 
                    if all(p['post_id'] != sp['post_id'] for sp in selected)
                ]
                
                if remaining_posts:
                    remaining_sample = random.sample(
                        remaining_posts,
                        min(remaining, len(remaining_posts))
                    )
                    selected.extend(remaining_sample)
        
        # 添加来源标记
        for post in selected:
            post['pool'] = self.name
            
        return selected

class ExposurePoolManager:
    """曝光池管理器，管理多个曝光池"""
    def __init__(self, db_manager=None):
        self.logger = logging.getLogger("exposure_pool_manager")
        self.db_manager = db_manager
        
        # 创建各曝光池
        self.pools = {
            "new": NewPostsPool(capacity=1000, weight=0.5),
            "hot": HotPostsPool(capacity=500, weight=0.3),
            "quality": QualityPostsPool(capacity=300, weight=0.2),
            "diversity": DiversityPostsPool(capacity=200, weight=0.1)
        }
        
        # 用户已曝光记录缓存 {user_id: {post_id: timestamp}}
        self.user_exposure_history = {}
        
        # 最近一次刷新时间
        self.last_refresh_time = None
        
    def refresh_pools(self, posts_df: Optional[pd.DataFrame] = None) -> Dict[str, int]:
        """刷新所有曝光池
        
        Args:
            posts_df: 帖子数据，如果为None则从数据库获取
        
        Returns:
            Dict[str, int]: 各池刷新的帖子数量
        """
        # 如果没有提供数据，尝试从数据库获取
        if posts_df is None and self.db_manager is not None:
            self.logger.info("从数据库获取帖子数据...")
            posts_df = self.db_manager.get_posts_data()
            
        if posts_df is None or len(posts_df) == 0:
            self.logger.error("无法获取帖子数据，曝光池刷新失败")
            return {}
            
        # 确保有必要的字段
        required_fields = ['post_id', 'heat_score', 'created_at']
        missing_fields = [f for f in required_fields if f not in posts_df.columns]
        
        if missing_fields:
            self.logger.error(f"帖子数据缺少必要字段: {missing_fields}")
            return {}
            
        # 刷新各池
        refresh_counts = {}
        for name, pool in self.pools.items():
            try:
                count = pool.refresh(posts_df)
                refresh_counts[name] = count
            except Exception as e:
                self.logger.error(f"刷新曝光池 {name} 失败: {e}")
                refresh_counts[name] = 0
                
        self.last_refresh_time = datetime.now()
        self.logger.info(f"曝光池刷新完成: {refresh_counts}")
        return refresh_counts
        
    def get_exposure_posts(self, user_id: int, count: int = 10, 
                          pool_distribution: Dict[str, float] = None) -> List[Dict]:
        """获取推荐曝光帖子
        
        Args:
            user_id: 用户ID
            count: 需要的帖子数量
            pool_distribution: 各池分配比例，如果为None则使用池的默认权重
            
        Returns:
            List[Dict]: 推荐帖子列表
        """
        # 获取用户历史曝光
        user_history = self.get_user_exposure_history(user_id)
        exclude_ids = list(user_history.keys())
        
        # 如果没有指定分配比例，使用池的默认权重
        if pool_distribution is None:
            pool_distribution = {name: pool.weight for name, pool in self.pools.items()}
            
            # 归一化权重
            total_weight = sum(pool_distribution.values())
            if total_weight > 0:
                pool_distribution = {
                    name: weight / total_weight 
                    for name, weight in pool_distribution.items()
                }
        
        # 计算每个池应该提供的帖子数量
        pool_counts = {}
        remaining = count
        
        for name, ratio in pool_distribution.items():
            if name not in self.pools:
                continue
                
            # 计算该池应提供的帖子数量
            pool_count = max(1, int(count * ratio))
            
            # 防止总数超过要求
            if sum(pool_counts.values()) + pool_count > count:
                pool_count = count - sum(pool_counts.values())
                
            if pool_count > 0:
                pool_counts[name] = pool_count
                remaining -= pool_count
                
        # 如果还有剩余配额，分配给有权重的池
        if remaining > 0 and pool_distribution:
            sorted_pools = sorted(
                pool_distribution.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for name, _ in sorted_pools:
                if name in self.pools and remaining > 0:
                    pool_counts[name] = pool_counts.get(name, 0) + 1
                    remaining -= 1
                    
                if remaining == 0:
                    break
        
        # 从各池获取帖子
        results = []
        for name, req_count in pool_counts.items():
            if name in self.pools and req_count > 0:
                pool_posts = self.pools[name].get_posts(
                    count=req_count, 
                    exclude_ids=exclude_ids
                )
                
                if pool_posts:
                    # 更新排除列表
                    exclude_ids.extend([p['post_id'] for p in pool_posts])
                    results.extend(pool_posts)
        
        # 如果结果不足，尝试从所有池中获取更多
        if len(results) < count:
            needed = count - len(results)
            self.logger.info(f"结果不足，尝试获取额外{needed}条帖子")
            
            for name, pool in self.pools.items():
                if len(results) >= count:
                    break
                    
                extra_posts = pool.get_posts(
                    count=needed, 
                    exclude_ids=exclude_ids
                )
                
                if extra_posts:
                    # 更新排除列表
                    exclude_ids.extend([p['post_id'] for p in extra_posts])
                    results.extend(extra_posts)
                    needed = count - len(results)
        
        # 记录曝光历史
        self._record_exposures(user_id, results)
        
        # 返回结果
        return results[:count]  # 确保不超过请求数量
    
    def _record_exposures(self, user_id: int, posts: List[Dict]) -> None:
        """记录用户曝光历史"""
        now = datetime.now()
        
        # 确保用户有历史记录字典
        if user_id not in self.user_exposure_history:
            self.user_exposure_history[user_id] = {}
            
        # 记录本次曝光
        for post in posts:
            post_id = post.get('post_id')
            if post_id:
                self.user_exposure_history[user_id][post_id] = now
                
        # 异步写入数据库（如果有数据库管理器）
        if self.db_manager is not None:
            try:
                exposure_records = [
                    {
                        'user_id': user_id,
                        'post_id': post.get('post_id'),
                        'exposure_time': now,
                        'pool': post.get('pool', 'unknown')
                    }
                    for post in posts if post.get('post_id')
                ]
                
                if exposure_records:
                    self.db_manager.record_exposures(exposure_records)
            except Exception as e:
                self.logger.error(f"记录曝光历史到数据库失败: {e}")
    
    def get_user_exposure_history(self, user_id: int, max_age_hours: int = 72) -> Dict[int, datetime]:
        """获取用户曝光历史
        
        Args:
            user_id: 用户ID
            max_age_hours: 最大历史时间，单位小时
            
        Returns:
            Dict[int, datetime]: 帖子ID到曝光时间的映射
        """
        # 从内存缓存中获取
        if user_id in self.user_exposure_history:
            history = self.user_exposure_history[user_id]
        else:
            history = {}
            
        # 如果有数据库，也从数据库获取
        if self.db_manager is not None and not history:
            try:
                db_history = self.db_manager.get_user_exposures(
                    user_id, 
                    hours=max_age_hours
                )
                
                if db_history:
                    for row in db_history:
                        post_id = row.get('post_id')
                        exp_time = row.get('exposure_time')
                        if post_id and exp_time:
                            history[post_id] = exp_time
                    
                    # 更新内存缓存
                    self.user_exposure_history[user_id] = history
            except Exception as e:
                self.logger.error(f"从数据库获取用户曝光历史失败: {e}")
                
        # 过滤掉过期的记录
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        history = {
            post_id: exp_time 
            for post_id, exp_time in history.items() 
            if exp_time > cutoff_time
        }
        
        return history
        
    def cleanup_exposure_history(self, max_age_hours: int = 72) -> int:
        """清理过期的曝光历史记录
        
        Returns:
            int: 清理的记录数
        """
        count = 0
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # 清理内存中的过期记录
        for user_id in list(self.user_exposure_history.keys()):
            history = self.user_exposure_history[user_id]
            expired = [
                post_id 
                for post_id, exp_time in history.items() 
                if exp_time < cutoff_time
            ]
            
            for post_id in expired:
                del history[post_id]
                count += 1
                
            # 如果用户没有任何记录，删除该用户的条目
            if not history:
                del self.user_exposure_history[user_id]
                
        # 如果有数据库，也清理数据库中的过期记录
        if self.db_manager is not None:
            try:
                db_count = self.db_manager.cleanup_exposures(hours=max_age_hours)
                self.logger.info(f"从数据库中清理了{db_count}条过期曝光记录")
            except Exception as e:
                self.logger.error(f"清理数据库曝光记录失败: {e}")
                
        self.logger.info(f"从内存中清理了{count}条过期曝光记录")
        return count
        
    def get_pool_stats(self) -> Dict[str, Dict]:
        """获取各曝光池统计信息"""
        stats = {}
        
        for name, pool in self.pools.items():
            stats[name] = {
                'name': name,
                'capacity': pool.capacity,
                'current_size': len(pool),
                'weight': pool.weight,
                'fill_ratio': len(pool) / pool.capacity if pool.capacity > 0 else 0
            }
            
        stats['last_refresh'] = self.last_refresh_time.isoformat() if self.last_refresh_time else None
        
        return stats 