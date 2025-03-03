import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import time
from threading import Lock
from .cache_manager import HotTopicsCacheManager

class HotTopicGenerator:
    """热点话题生成器"""
    def __init__(self, db_manager=None, cache_manager=None):
        self.logger = logging.getLogger("hot_topic_generator")
        self.db_manager = db_manager
        self.cache_manager = cache_manager or HotTopicsCacheManager()
        
        # 参数配置
        self.top_n = 50  # 生成的热点数量
        self.time_window_hours = 24  # 时间窗口（小时）
        self.min_interactions = 5  # 最小交互数
        
        # 权重设置
        self.weights = {
            'view': 1.0,      # 浏览
            'like': 3.0,      # 点赞
            'collect': 4.0,   # 收藏
            'comment': 5.0,   # 评论
            'share': 4.0      # 分享
        }
        
        # 时间衰减系数（越新的内容权重越高）
        self.time_decay_factor = 0.8
        
        # 当前热点话题列表
        self.current_hot_topics = []
        
        # 上次生成时间
        self.last_generation_time = None
        
        # 线程安全锁
        self.lock = Lock()
        
    def generate_hot_topics(self, force: bool = False) -> List[Dict]:
        """生成热点话题
        
        Args:
            force: 是否强制重新生成
            
        Returns:
            List[Dict]: 热点话题列表
        """
        # 如果缓存中有有效数据，直接返回
        if not force and self.cache_manager:
            cached_data = self.cache_manager.get_hot_topics()
            if cached_data:
                self.logger.info("从缓存获取热点话题数据")
                self.current_hot_topics = cached_data.get('hot_topics', [])
                return self.current_hot_topics
                
        # 加锁确保线程安全
        with self.lock:
            self.logger.info("开始生成热点话题...")
            start_time = time.time()
            
            try:
                # 获取数据
                if self.db_manager is None:
                    self.logger.error("缺少数据库管理器，无法生成热点话题")
                    return self.current_hot_topics
                    
                # 获取时间窗口内的帖子数据
                posts_df = self._get_posts_data()
                
                if posts_df is None or len(posts_df) == 0:
                    self.logger.warning("未获取到帖子数据，使用当前热点话题")
                    return self.current_hot_topics
                    
                # 获取时间窗口内的用户交互数据
                interactions_df = self._get_interaction_data()
                
                if interactions_df is None or len(interactions_df) == 0:
                    self.logger.warning("未获取到用户交互数据，使用当前热点话题")
                    return self.current_hot_topics
                    
                # 计算热度得分
                hot_posts = self._calculate_heat_scores(posts_df, interactions_df)
                
                if hot_posts is None or len(hot_posts) == 0:
                    self.logger.warning("热度计算结果为空，使用当前热点话题")
                    return self.current_hot_topics
                    
                # 格式化结果
                self.current_hot_topics = self._format_hot_topics(hot_posts)
                
                # 更新生成时间
                self.last_generation_time = datetime.now()
                
                # 保存到数据库
                self._save_to_database(self.current_hot_topics)
                
                # 缓存结果
                if self.cache_manager:
                    self.cache_manager.set_hot_topics(
                        self.current_hot_topics,
                        ttl=300  # 5分钟缓存
                    )
                    
                end_time = time.time()
                self.logger.info(
                    f"热点话题生成完成，共{len(self.current_hot_topics)}条，"
                    f"耗时: {end_time - start_time:.2f}秒"
                )
                
                return self.current_hot_topics
                
            except Exception as e:
                self.logger.error(f"生成热点话题时出错: {e}")
                return self.current_hot_topics
    
    def _get_posts_data(self) -> Optional[pd.DataFrame]:
        """获取帖子数据"""
        try:
            # 从数据库获取帖子数据
            posts_df = self.db_manager.get_posts_data()
            return posts_df
        except Exception as e:
            self.logger.error(f"获取帖子数据失败: {e}")
            return None
    
    def _get_interaction_data(self) -> Optional[pd.DataFrame]:
        """获取用户交互数据"""
        try:
            # 计算时间窗口
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=self.time_window_hours)
            
            # 从数据库获取用户交互数据
            interactions_df = self.db_manager.get_user_interactions(
                start_time=start_time,
                end_time=end_time
            )
            return interactions_df
        except Exception as e:
            self.logger.error(f"获取用户交互数据失败: {e}")
            return None
    
    def _calculate_heat_scores(self, posts_df: pd.DataFrame, 
                              interactions_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算热度分数
        
        Args:
            posts_df: 帖子数据
            interactions_df: 交互数据
            
        Returns:
            pd.DataFrame: 热度计算结果
        """
        try:
            # 确保数据有必要的列
            required_cols = ['post_id', 'interaction_type', 'created_at']
            if not all(col in interactions_df.columns for col in required_cols):
                self.logger.error(f"交互数据缺少必要的列: {required_cols}")
                return None
                
            # 确保交互数据有时间列并是日期时间类型
            if 'created_at' in interactions_df.columns:
                interactions_df['created_at'] = pd.to_datetime(interactions_df['created_at'])
            
            # 当前时间
            now = datetime.now()
            
            # 计算时间衰减
            interactions_df['time_decay'] = interactions_df['created_at'].apply(
                lambda x: self._calculate_time_decay(x, now)
            )
            
            # 应用交互类型权重
            interactions_df['weighted_score'] = interactions_df.apply(
                lambda row: self.weights.get(row['interaction_type'], 1.0) * row['time_decay'],
                axis=1
            )
            
            # 按帖子ID分组汇总
            grouped = interactions_df.groupby('post_id').agg({
                'weighted_score': 'sum',
                'interaction_type': 'count'
            }).reset_index()
            
            # 重命名列
            grouped = grouped.rename(columns={
                'weighted_score': 'heat_score',
                'interaction_type': 'interaction_count'
            })
            
            # 过滤掉交互数低于阈值的
            grouped = grouped[grouped['interaction_count'] >= self.min_interactions]
            
            # 合并帖子信息
            if 'post_id' in posts_df.columns:
                hot_posts = pd.merge(
                    grouped,
                    posts_df,
                    on='post_id',
                    how='inner'
                )
            else:
                hot_posts = grouped
                
            # 按热度得分降序排序
            hot_posts = hot_posts.sort_values('heat_score', ascending=False)
            
            # 保留前N个
            hot_posts = hot_posts.head(self.top_n)
            
            return hot_posts
        except Exception as e:
            self.logger.error(f"计算热度分数失败: {e}")
            return None
    
    def _calculate_time_decay(self, timestamp: datetime, now: datetime) -> float:
        """计算时间衰减因子
        
        Args:
            timestamp: 交互时间戳
            now: 当前时间
            
        Returns:
            float: 时间衰减权重
        """
        # 计算时间差（小时）
        delta_hours = (now - timestamp).total_seconds() / 3600
        
        # 使用指数衰减
        decay = np.exp(-self.time_decay_factor * delta_hours / self.time_window_hours)
        
        return decay
    
    def _format_hot_topics(self, hot_posts: pd.DataFrame) -> List[Dict]:
        """格式化热点话题结果
        
        Args:
            hot_posts: 热点帖子数据
            
        Returns:
            List[Dict]: 格式化的热点话题列表
        """
        hot_topics = []
        
        for idx, row in hot_posts.iterrows():
            try:
                topic = {
                    'rank': idx + 1,
                    'post_id': int(row['post_id']),
                    'heat_score': float(row['heat_score']),
                    'title': row.get('title', ''),
                    'interaction_count': int(row['interaction_count']),
                    'timestamp': datetime.now().isoformat()
                }
                
                # 添加可能存在的其他字段
                for field in ['author_id', 'category', 'created_at', 'summary']:
                    if field in row and not pd.isna(row[field]):
                        if field == 'created_at':
                            topic[field] = row[field].isoformat()
                        else:
                            topic[field] = row[field]
                            
                hot_topics.append(topic)
            except Exception as e:
                self.logger.error(f"格式化热点话题失败: {e}, 行数据: {row}")
                
        return hot_topics
    
    def _save_to_database(self, hot_topics: List[Dict]) -> bool:
        """保存热点话题到数据库
        
        Args:
            hot_topics: 热点话题列表
            
        Returns:
            bool: 是否保存成功
        """
        if not hot_topics or self.db_manager is None:
            return False
            
        try:
            # 更新当前热点表
            self.db_manager.update_current_hot_topics(hot_topics)
            
            # 保存到历史记录表
            self.db_manager.save_hot_topics_history(hot_topics)
            
            return True
        except Exception as e:
            self.logger.error(f"保存热点话题到数据库失败: {e}")
            return False
    
    def get_hot_topics(self, count: int = None, force_refresh: bool = False) -> List[Dict]:
        """获取热点话题
        
        Args:
            count: 需要的数量，None表示全部
            force_refresh: 是否强制刷新
            
        Returns:
            List[Dict]: 热点话题列表
        """
        # 检查是否需要刷新
        if force_refresh or not self.current_hot_topics:
            self.generate_hot_topics(force=force_refresh)
            
        # 返回指定数量
        if count is not None and count > 0:
            return self.current_hot_topics[:count]
        else:
            return self.current_hot_topics
    
    def get_last_generation_time(self) -> Optional[datetime]:
        """获取上次生成时间"""
        return self.last_generation_time 