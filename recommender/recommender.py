# -*- coding: utf-8 -*-
"""
推荐算法实现
"""
import numpy as np
import pandas as pd
import joblib
import json
import time
import random
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from database.database import get_db_pool, get_db_manager
from cache import memory_cache, disk_cache, cached_get, cache_result, TimerContext
from utils.monitor import track_performance, performance_collector
import logging
from typing import List, Dict, Any
from config.config_manager import get_config_manager
from cache.cache_manager import get_cache_manager

logger = logging.getLogger(__name__)

class RecommenderConfig:
    """推荐系统配置管理"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(RecommenderConfig, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if getattr(self, '_initialized', False):
            return
            
        with self._lock:
            self.logger = logging.getLogger("recommender_config")
            self.config_manager = get_config_manager()
            self.logger.info("开始初始化推荐系统配置...")
            
            try:
                self._load_config()
                # 使用register_observer替代add_listener
                self.config_manager.register_observer(self)
                self._initialized = True
                self.logger.info("推荐系统配置初始化完成")
            except Exception as e:
                self.logger.error(f"推荐系统配置初始化失败: {str(e)}")
                raise
    
    def _load_config(self):
        """加载配置"""
        try:
            self.logger.debug("开始加载推荐系统配置...")
            recommender_config = self.config_manager.get('recommender', {})
            
            # TFIDF参数配置
            self.tfidf_params = {
                'max_features': recommender_config.get('tfidf_max_features', 5000),
                'min_df': recommender_config.get('tfidf_min_df', 2),
                'max_df': recommender_config.get('tfidf_max_df', 0.95),
                'ngram_range': tuple(recommender_config.get('tfidf_ngram_range', (1, 2)))
            }
            self.logger.debug(f"已加载TFIDF参数配置: {self.tfidf_params}")
            
            # 行为权重配置
            self.behavior_weights = recommender_config.get('behavior_weights', {
                'view': 1,
                'click': 2,
                'like': 3,
                'collect': 4,
                'comment': 4,
                'comment_like': 2
            })
            self.logger.debug(f"已加载行为权重配置: {self.behavior_weights}")
            
            # 其他配置...
            self.min_interaction_threshold = recommender_config.get('min_interaction_threshold', 5)
            self.max_recommendations = recommender_config.get('max_recommendations', 100)
            self.default_page_size = recommender_config.get('default_page_size', 20)
            
            self.logger.info("推荐系统配置加载完成")
        except Exception as e:
            self.logger.error(f"加载推荐系统配置失败: {str(e)}")
            raise
    
    def config_updated(self, path: str, new_value: Any):
        """配置变更处理"""
        if path.startswith('recommender.'):
            self.logger.info(f"检测到推荐系统配置变更: {path}")
            with self._lock:
                try:
                    self._load_config()
                    self.logger.info("推荐系统配置已更新")
                except Exception as e:
                    self.logger.error(f"更新推荐系统配置失败: {str(e)}")
    
    @property
    def TFIDF_PARAMS(self) -> Dict:
        return self.tfidf_params
    
    @property
    def BEHAVIOR_WEIGHTS(self) -> Dict:
        return self.behavior_weights

# 获取配置实例
config = RecommenderConfig()

# 初始化数据库连接池
db_pool = get_db_pool()

# 定义热点话题配置
HOT_TOPICS_CONFIG = {
    'max_topics': 50,  # 默认值
    'min_score': 0.1,  # 默认值
    'time_window': timedelta(days=7)  # 默认值
}

# 尝试从配置中更新值
try:
    HOT_TOPICS_CONFIG.update({
        'max_topics': config.config_manager.get('hot_topics_max', 50),
        'min_score': config.config_manager.get('hot_topics_min_score', 0.1),
        'time_window': timedelta(days=config.config_manager.get('hot_topics_window_days', 7))
    })
except Exception as e:
    logger.warning(f"加载热点话题配置失败，使用默认值: {str(e)}")

# 并行处理的工作线程数
MAX_WORKERS = 4  # 默认值
try:
    MAX_WORKERS = config.config_manager.get('max_workers', 4)
except Exception as e:
    logger.warning(f"加载最大工作线程配置失败，使用默认值: {str(e)}")

class Recommender:
    """推荐系统核心类
    
    负责生成和管理推荐结果
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Recommender, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        with self._lock:
            if self._initialized:
                return
                
            self.logger = logging.getLogger("recommender")
            self.config_manager = get_config_manager()
            self.db_manager = get_db_manager()
            self.cache_manager = get_cache_manager()
            
            # 获取配置
            self.config = self.config_manager.get('recommender', {})
            self.exposure_config = self.config_manager.get('exposure', {})
            
            # 注册为配置观察者
            self.config_manager.register_observer(self)
            
            self._initialized = True
            self.logger.info("推荐器初始化完成")
    
    def config_updated(self, path: str, new_value: Any):
        """配置更新回调"""
        if path.startswith('recommender.') or path.startswith('exposure.'):
            self.logger.info(f"检测到推荐配置变更: {path}")
            # 更新本地配置
            self.config = self.config_manager.get('recommender')
            self.exposure_config = self.config_manager.get('exposure')
            # 清除相关缓存
            self.cache_manager.clear_pattern('recommendations:*')
    
    def get_recommendations(self, user_id: int, page: int = 1, page_size: int = None) -> List[Dict]:
        """获取用户推荐
        
        Args:
            user_id: 用户ID
            page: 页码
            page_size: 每页数量
            
        Returns:
            List[Dict]: 推荐结果列表
        """
        # 使用配置的默认页面大小
        if page_size is None:
            page_size = self.config['default_page_size']
        
        # 限制页面大小
        page_size = min(page_size, self.config['max_recommendations'])
        
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 尝试从缓存获取
        cache_key = f'recommendations:user:{user_id}:page:{page}'
        recommendations = self.cache_manager.get(cache_key)
        if recommendations is not None:
            return recommendations
            
        # 生成推荐
        recommendations = self._generate_recommendations(user_id, offset, page_size)
        
        # 缓存结果
        cache_ttl = self.config_manager.get('cache.ttl.user_recommendations')
        self.cache_manager.set(cache_key, recommendations, ttl=cache_ttl)
        
        return recommendations
    
    def _generate_recommendations(self, user_id: int, offset: int, limit: int) -> List[Dict]:
        """生成推荐结果
        
        Args:
            user_id: 用户ID
            offset: 偏移量
            limit: 限制数量
            
        Returns:
            List[Dict]: 推荐结果列表
        """
        # 获取算法权重
        weights = self.config['algorithm_weights']
        
        # 获取用户行为权重
        behavior_weights = self.config['behavior_weights']
        
        # 获取时间衰减参数
        time_decay = self.config['time_decay']
        
        # 基于内容的推荐
        content_based = self._get_content_based_recommendations(
            user_id, 
            limit=int(limit * weights['content_based'])
        )
        
        # 协同过滤推荐
        collaborative = self._get_collaborative_recommendations(
            user_id,
            limit=int(limit * weights['collaborative'])
        )
        
        # 热点推荐
        hot_topics = self._get_hot_topics_recommendations(
            user_id,
            limit=int(limit * weights['hot_topics'])
        )
        
        # 合并结果
        recommendations = []
        recommendations.extend(content_based)
        recommendations.extend(collaborative)
        recommendations.extend(hot_topics)
        
        # 根据分数排序
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # 应用曝光控制
        recommendations = self._apply_exposure_control(recommendations)
        
        # 返回分页结果
        return recommendations[offset:offset + limit]
    
    def _get_content_based_recommendations(self, user_id: int, limit: int) -> List[Dict]:
        """获取基于内容的推荐"""
        # TODO: 实现基于内容的推荐算法
        return []
    
    def _get_collaborative_recommendations(self, user_id: int, limit: int) -> List[Dict]:
        """获取协同过滤推荐"""
        # TODO: 实现协同过滤推荐算法
        return []
    
    def _get_hot_topics_recommendations(self, user_id: int, limit: int) -> List[Dict]:
        """获取热点推荐"""
        # TODO: 实现热点推荐算法
        return []
    
    def _apply_exposure_control(self, recommendations: List[Dict]) -> List[Dict]:
        """应用曝光控制
        
        根据曝光池配置过滤和调整推荐结果
        
        Args:
            recommendations: 原始推荐结果
            
        Returns:
            List[Dict]: 调整后的推荐结果
        """
        # 获取全局曝光比例
        global_ratio = self.exposure_config.get('global_ratio', 0.8)
        
        # 获取各池子配置
        pools = self.exposure_config.get('pools', {})
        
        # 应用曝光控制逻辑
        controlled_recommendations = []
        
        # TODO: 实现具体的曝光控制逻辑
        
        return controlled_recommendations or recommendations

    def generate_hot_topics(self, time_window: timedelta = None) -> List[Dict]:
        """生成热点话题
        
        Args:
            time_window: 时间窗口，默认使用配置中的值
            
        Returns:
            List[Dict]: 热点话题列表，每个话题包含id、title、score等信息
        """
        if time_window is None:
            time_window = HOT_TOPICS_CONFIG['time_window']
            
        self.logger.info(f"开始生成热点话题，时间窗口：{time_window}")
        
        try:
            # 计算时间范围
            end_time = datetime.now()
            start_time = end_time - time_window
            
            # 查询热门帖子
            sql = """
            SELECT 
                p.id,
                p.title,
                p.content,
                p.created_at,
                COUNT(DISTINCT v.user_id) as view_count,
                COUNT(DISTINCT l.user_id) as like_count,
                COUNT(DISTINCT c.id) as comment_count
            FROM posts p
            LEFT JOIN post_views v ON p.id = v.post_id AND v.created_at BETWEEN %s AND %s
            LEFT JOIN post_likes l ON p.id = l.post_id AND l.created_at BETWEEN %s AND %s
            LEFT JOIN comments c ON p.id = c.post_id AND c.created_at BETWEEN %s AND %s
            WHERE p.created_at BETWEEN %s AND %s
            GROUP BY p.id
            HAVING (view_count + like_count * 2 + comment_count * 3) >= %s
            ORDER BY (view_count + like_count * 2 + comment_count * 3) DESC
            LIMIT %s
            """
            
            params = (
                start_time, end_time,  # views
                start_time, end_time,  # likes
                start_time, end_time,  # comments
                start_time, end_time,  # posts
                HOT_TOPICS_CONFIG['min_score'],
                HOT_TOPICS_CONFIG['max_topics']
            )
            
            hot_posts = self.db_manager.execute_query(sql, params)
            
            # 计算热度分数并格式化结果
            topics = []
            for post in hot_posts:
                score = (
                    post['view_count'] + 
                    post['like_count'] * 2 + 
                    post['comment_count'] * 3
                )
                
                topics.append({
                    'id': post['id'],
                    'title': post['title'],
                    'content': post['content'][:200],  # 截取前200个字符
                    'score': score,
                    'view_count': post['view_count'],
                    'like_count': post['like_count'],
                    'comment_count': post['comment_count'],
                    'created_at': post['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                })
            
            self.logger.info(f"热点话题生成完成，共{len(topics)}个话题")
            return topics
            
        except Exception as e:
            self.logger.error(f"生成热点话题失败: {str(e)}")
            return []

    def batch_generate_recommendations(self, batch_size: int = 100) -> None:
        """批量生成用户推荐
        
        Args:
            batch_size: 每批处理的用户数量
        """
        self.logger.info(f"开始批量生成推荐，批次大小：{batch_size}")
        
        try:
            # 获取所有活跃用户
            sql = """
            SELECT DISTINCT user_id 
            FROM user_behaviors 
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            """
            users = self.db_manager.execute_query(sql)
            
            if not users:
                self.logger.warning("没有找到活跃用户")
                return
                
            total_users = len(users)
            self.logger.info(f"找到 {total_users} 个活跃用户")
            
            # 创建线程池
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # 分批处理用户
                for i in range(0, total_users, batch_size):
                    batch_users = users[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    total_batches = (total_users + batch_size - 1) // batch_size
                    
                    self.logger.info(f"处理第 {batch_num}/{total_batches} 批用户 ({len(batch_users)} 个用户)")
                    
                    # 并行生成推荐
                    futures = []
                    for user in batch_users:
                        future = executor.submit(
                            self._generate_and_cache_recommendations,
                            user['user_id']
                        )
                        futures.append(future)
                    
                    # 等待当前批次完成
                    for future in futures:
                        try:
                            future.result()
                        except Exception as e:
                            self.logger.error(f"生成推荐失败: {str(e)}")
                    
                    self.logger.info(f"第 {batch_num}/{total_batches} 批处理完成")
            
            self.logger.info("批量生成推荐完成")
            
        except Exception as e:
            self.logger.error(f"批量生成推荐失败: {str(e)}")
            raise
    
    def _generate_and_cache_recommendations(self, user_id: int) -> None:
        """为单个用户生成推荐并缓存
        
        Args:
            user_id: 用户ID
        """
        try:
            # 生成推荐
            recommendations = self._generate_recommendations(
                user_id,
                offset=0,
                limit=self.config.get('max_recommendations', 100)
            )
            
            # 缓存推荐结果
            cache_key = f'recommendations:user:{user_id}:page:1'
            cache_ttl = self.config_manager.get('cache.ttl.user_recommendations', 3600)
            self.cache_manager.set(cache_key, recommendations, ttl=cache_ttl)
            
        except Exception as e:
            self.logger.error(f"为用户 {user_id} 生成推荐失败: {str(e)}")
            raise

# 全局推荐器实例
_recommender = None

def get_recommender() -> Recommender:
    """获取推荐器实例
    
    Returns:
        Recommender: 推荐器实例
    """
    global _recommender
    if _recommender is None:
        _recommender = Recommender()
    return _recommender

class FeatureProcessor:
    """特征处理器
    
    负责处理文本特征和用户画像生成
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(FeatureProcessor, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if getattr(self, '_initialized', False):
            return
            
        with self._lock:
            try:
                self.logger = logging.getLogger("feature_processor")
                self.logger.info("开始初始化特征处理器...")
                
                self.config_manager = get_config_manager()
                self.cache_manager = get_cache_manager()
                
                # 获取配置
                self.config = self.config_manager.get('recommender', {})
                
                # 初始化推荐配置
                self.recommender_config = RecommenderConfig()
                
                # 初始化TF-IDF向量器
                self.logger.debug("初始化TF-IDF向量器...")
                self.tfidf = TfidfVectorizer(**self.recommender_config.TFIDF_PARAMS)
                self.feature_names = None
                self.posts = None
                self.tfidf_matrix = None
                self.svd_model = None
                self.svd_features = None
                
                # 注册为配置观察者
                self.config_manager.register_observer(self)
                
                self._initialized = True
                self.logger.info("特征处理器初始化完成")
            except Exception as e:
                self.logger.error(f"特征处理器初始化失败: {str(e)}")
                raise
    
    def load_data(self):
        """加载帖子数据"""
        self.logger.info("开始加载帖子数据...")
        try:
            with self._lock:
                # 先尝试从缓存读取
                self.logger.debug("尝试从缓存读取帖子数据...")
                cached_posts = disk_cache.get('posts_data')
                if cached_posts is not None:
                    self.posts = cached_posts
                    self.logger.info(f"从缓存加载了 {len(self.posts)} 条帖子数据")
                    return
                
                # 从posts表加载帖子数据
                self.logger.debug("从数据库加载帖子数据...")
                sql = """
                SELECT p.post_id, p.post_title as title, p.hashtags as tags,
                       COALESCE(GROUP_CONCAT(DISTINCT c.content SEPARATOR ' '), '') as content
                FROM posts p
                LEFT JOIN comments c ON p.post_id = c.post_id
                GROUP BY p.post_id, p.post_title, p.hashtags
                """
                
                results = db_pool.query(sql)
                if not results:
                    self.logger.warning("没有找到任何帖子数据")
                    return
                
                # 转换结果为DataFrame
                self.posts = pd.DataFrame(results)
                self.posts['tags'] = self.posts['tags'].fillna('').str.replace(',', ' ')
                
                # 合并标签和评论内容作为特征
                self.logger.debug("处理帖子特征...")
                self.posts['features'] = self.posts.apply(
                    lambda x: x['tags'] + ' ' + ' '.join(x['content'].split()[:50]) 
                    if x['content'] else x['tags'], axis=1
                )
                
                # 缓存结果
                self.logger.debug("缓存帖子数据...")
                disk_cache.set('posts_data', self.posts)
                
                self.logger.info(f"成功加载并处理了 {len(self.posts)} 条帖子数据")
                
                # 如果已有模型就加载，否则训练
                try:
                    self.logger.debug("尝试加载已有的TF-IDF模型...")
                    self.tfidf = joblib.load('models/tfidf_model.pkl')
                    self.feature_names = self.tfidf.get_feature_names_out()
                    self.logger.info("成功加载已有的TF-IDF模型")
                    
                    # 尝试加载SVD模型
                    try:
                        self.svd_model = joblib.load('models/svd_model.pkl')
                        self.logger.info("成功加载SVD模型")
                    except Exception as e:
                        self.logger.warning(f"加载SVD模型失败: {str(e)}")
                except Exception as e:
                    self.logger.info(f"加载TF-IDF模型失败，将重新训练: {str(e)}")
                    self.fit_model()
                    
        except Exception as e:
            self.logger.error(f"加载帖子数据失败: {str(e)}")
            raise
    
    def fit_model(self):
        """训练TF-IDF模型"""
        self.logger.info("开始训练TF-IDF模型...")
        try:
            with self._lock:
                if self.posts is None or len(self.posts) == 0:
                    self.logger.debug("没有帖子数据，尝试加载...")
                    self.load_data()
                
                if not self.posts.empty:
                    # 数据预处理
                    self.logger.debug("开始数据预处理...")
                    feature_texts = self.posts['features'].fillna('')
                    
                    # 文本清理和验证
                    valid_texts = []
                    invalid_count = 0
                    for text in feature_texts:
                        # 转换为字符串并清理
                        text = str(text).strip()
                        # 移除过短的文本
                        if len(text) >= 2:  # 至少2个字符
                            # 检查是否只包含停用词或无意义字符
                            words = text.split()
                            if len(words) > 0:
                                valid_texts.append(text)
                            else:
                                invalid_count += 1
                        else:
                            invalid_count += 1
                    
                    self.logger.info(f"文本预处理完成: 有效文本 {len(valid_texts)} 条，无效文本 {invalid_count} 条")
                    
                    # 如果没有足够的有效文本，使用默认模型
                    if len(valid_texts) < 2:
                        self.logger.warning(f"有效文本数量不足(当前: {len(valid_texts)})，使用默认模型")
                        # 创建一个基础的TF-IDF矩阵
                        self.tfidf = TfidfVectorizer(**self.recommender_config.TFIDF_PARAMS)
                        self.tfidf_matrix = self.tfidf.fit_transform(['default text placeholder'])
                        self.feature_names = self.tfidf.get_feature_names_out()
                        return
                    
                    self.logger.info(f"使用 {len(valid_texts)} 条有效文本进行训练")
                    
                    # 配置TF-IDF向量器
                    tfidf_params = {
                        'max_features': 5000,  # 限制特征数量
                        'min_df': 1,  # 词必须至少出现1次
                        'max_df': 0.95,  # 出现在超过95%文档中的词会被移除
                        'ngram_range': (1, 2),  # 使用单个词和双词组合
                        'stop_words': None,  # 不使用停用词，因为可能有中文内容
                        'strip_accents': 'unicode',
                        'lowercase': True,
                        'analyzer': 'word',
                        'token_pattern': r'(?u)\b\w+\b'  # 匹配任何单词字符
                    }
                    
                    # 使用配置中的参数覆盖默认值
                    if hasattr(self, 'recommender_config'):
                        tfidf_params.update(self.recommender_config.TFIDF_PARAMS)
                    
                    self.logger.debug(f"TF-IDF参数配置: {tfidf_params}")
                    
                    # 训练模型
                    self.tfidf = TfidfVectorizer(**tfidf_params)
                    
                    try:
                        self.tfidf_matrix = self.tfidf.fit_transform(valid_texts)
                        self.feature_names = self.tfidf.get_feature_names_out()
                        
                        self.logger.info(f"TF-IDF模型训练完成，特征数量: {len(self.feature_names)}")
                        
                        # 确保models目录存在
                        import os
                        os.makedirs('models', exist_ok=True)
                        
                        # 保存模型
                        try:
                            self.logger.debug("保存TF-IDF模型...")
                            joblib.dump(self.tfidf, 'models/tfidf_model.pkl')
                            self.logger.info("TF-IDF模型保存成功")
                        except Exception as e:
                            self.logger.warning(f"保存TF-IDF模型失败: {str(e)}")
                    except ValueError as ve:
                        self.logger.warning(f"模型训练出现问题，使用默认模型: {str(ve)}")
                        # 创建一个基础的TF-IDF矩阵
                        self.tfidf = TfidfVectorizer(**tfidf_params)
                        self.tfidf_matrix = self.tfidf.fit_transform(['default text placeholder'])
                        self.feature_names = self.tfidf.get_feature_names_out()
                else:
                    self.logger.warning("没有可用的训练数据，使用默认模型")
                    # 创建一个基础的TF-IDF矩阵
                    self.tfidf = TfidfVectorizer(**self.recommender_config.TFIDF_PARAMS)
                    self.tfidf_matrix = self.tfidf.fit_transform(['default text placeholder'])
                    self.feature_names = self.tfidf.get_feature_names_out()
                    
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            # 确保即使出错也有一个可用的基础模型
            self.tfidf = TfidfVectorizer(**self.recommender_config.TFIDF_PARAMS)
            self.tfidf_matrix = self.tfidf.fit_transform(['default text placeholder'])
            self.feature_names = self.tfidf.get_feature_names_out()
    
    def config_updated(self, path: str, new_value: Any):
        """配置更新回调"""
        if path.startswith('recommender.'):
            self.logger.info(f"检测到推荐配置变更: {path}")
            with self._lock:
                # 更新本地配置
                self.config = self.config_manager.get('recommender', {})
                # 清除相关缓存
                self.cache_manager.clear_pattern('features:*')
                # 重新训练模型
                try:
                    self.fit_model()
                    self.logger.info("配置更新后模型重训练成功")
                except Exception as e:
                    self.logger.error(f"配置更新后模型重训练失败: {str(e)}")
    
    @track_performance('get_tfidf_matrix', performance_collector)
    def get_tfidf_matrix(self):
        """获取TF-IDF矩阵（带缓存）"""
        try:
            with self._lock:
                if self.tfidf_matrix is not None:
                    return self.tfidf_matrix
                    
                if self.posts is None:
                    self.load_data()
                    
                if self.posts is not None and not self.posts.empty:
                    # 计算TF-IDF矩阵
                    feature_texts = self.posts['features'].fillna('')
                    self.tfidf_matrix = self.tfidf.transform(feature_texts)
                    return self.tfidf_matrix
                else:
                    self.logger.warning("无法获取TF-IDF矩阵：没有可用的帖子数据")
                    return None
        except Exception as e:
            self.logger.error(f"获取TF-IDF矩阵失败: {str(e)}")
            raise
    
    @track_performance('get_user_profile', performance_collector)
    def get_user_profile(self, user_id):
        """生成用户兴趣向量（带缓存）"""
        # 尝试从缓存获取
        cache_key = f"user_profile_{user_id}"
        cached_profile = memory_cache.get(cache_key)
        if cached_profile is not None:
            return cached_profile
        
        # 使用UNION ALL一次性获取所有用户行为数据，减少数据库连接次数
        sql = """
        SELECT p.post_id, behavior, p.hashtags as tags
        FROM (
            -- 浏览行为
            SELECT v.post_id, 'view' as behavior, v.user_id 
            FROM user_views v
            WHERE v.user_id = %s 
              AND v.created_at > DATE_SUB(NOW(), INTERVAL %s DAY)
              AND v.post_id IS NOT NULL
            
            UNION ALL
            
            -- 点赞行为
            SELECT l.post_id, 'like' as behavior, l.user_id
            FROM post_likes l
            WHERE l.user_id = %s 
              AND l.created_at > DATE_SUB(NOW(), INTERVAL %s DAY)
            
            UNION ALL
            
            -- 收藏行为
            SELECT c.post_id, 'collect' as behavior, c.user_id
            FROM post_collects c
            WHERE c.user_id = %s 
              AND c.created_at > DATE_SUB(NOW(), INTERVAL %s DAY)
            
            UNION ALL
            
            -- 评论行为
            SELECT c.post_id, 'comment' as behavior, c.user_id
            FROM comments c
            WHERE c.user_id = %s 
              AND c.created_at > DATE_SUB(NOW(), INTERVAL %s DAY)
            
            UNION ALL
            
            -- 评论点赞行为
            SELECT c.post_id, 'comment_like' as behavior, cl.user_id
            FROM comment_likes cl
            JOIN comments c ON cl.comment_id = c.comment_id
            WHERE cl.user_id = %s 
              AND cl.created_at > DATE_SUB(NOW(), INTERVAL %s DAY)
        ) as user_behaviors
        JOIN posts p ON user_behaviors.post_id = p.post_id
        """
        
        days = self.recommender_config.config_manager.get('recent_days', 7)
        params = (
            user_id, days,
            user_id, days,
            user_id, days,
            user_id, days,
            user_id, days
        )
        
        results = db_pool.query(sql, params)
        
        if not results:
            return None
        
        # 初始化用户向量
        if self.feature_names is None:
            self.load_data()
            
        if self.feature_names is None or len(self.feature_names) == 0:
            return None
            
        user_vector = np.zeros(len(self.feature_names))
        
        # 创建帖子标签到向量的映射缓存，避免重复转换
        tag_vectors = {}
        
        # 累加用户兴趣
        for row in results:
            behavior = row['behavior']
            tags = row['tags'].replace(',', ' ') if row['tags'] else ''
            
            if not tags:
                continue
            
            weight = config.BEHAVIOR_WEIGHTS.get(behavior, 1)
            try:
                # 检查缓存中是否已有此标签的向量
                if tags not in tag_vectors:
                    tag_vectors[tags] = self.tfidf.transform([tags]).toarray()[0]
                
                user_vector += tag_vectors[tags] * weight
            except Exception as e:
                print(f"处理用户{user_id}的兴趣向量失败: {str(e)}")
                continue
        
        # 归一化 - 添加防御性代码避免除以零
        norm = np.linalg.norm(user_vector)
        if norm < 1e-10:  # 使用小值阈值而不是精确的零
            return None
            
        result = user_vector / norm
        
        # 缓存结果（1小时过期）
        memory_cache.set(cache_key, result, 3600)
        
        return result
    
    def get_similar_users(self, user_id, limit=20):
        """找出相似用户"""
        # 获取用户画像
        user_profile = self.get_user_profile(user_id)
        if user_profile is None:
            return []
        
        # 批量获取其他用户画像 - 从活跃用户中查找
        sql = """
        SELECT DISTINCT user_id 
        FROM (
            SELECT user_id FROM post_likes
            WHERE user_id != %s AND created_at > DATE_SUB(NOW(), INTERVAL %s DAY)
            UNION
            SELECT user_id FROM post_collects
            WHERE user_id != %s AND created_at > DATE_SUB(NOW(), INTERVAL %s DAY)
            UNION
            SELECT user_id FROM user_views
            WHERE user_id != %s AND created_at > DATE_SUB(NOW(), INTERVAL %s DAY)
            UNION
            SELECT user_id FROM comments
            WHERE user_id != %s AND created_at > DATE_SUB(NOW(), INTERVAL %s DAY)
            LIMIT 1000
        ) as active_users
        """
        days = self.recommender_config.config_manager.get('recent_days', 7)
        users = db_pool.query(sql, (
            user_id, days, 
            user_id, days, 
            user_id, days,
            user_id, days
        ))
        
        if not users:
            return []
        
        # 计算相似度
        similarities = []
        for other_user in users:
            other_id = other_user['user_id']
            other_profile = self.get_user_profile(other_id)
            
            if other_profile is not None:
                try:
                    # 计算余弦相似度，添加防御性代码防止除零
                    norm_user = np.linalg.norm(user_profile)
                    norm_other = np.linalg.norm(other_profile)
                    
                    # 检查向量长度是否非零
                    if norm_user < 1e-10 or norm_other < 1e-10:
                        continue
                    
                    similarity = np.dot(user_profile, other_profile) / (norm_user * norm_other)
                    similarities.append((other_id, similarity))
                except Exception as e:
                    print(f"计算用户{user_id}和{other_id}的相似度失败: {str(e)}")
                    continue
        
        # 排序返回最相似的用户
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [user[0] for user in similarities[:limit]]