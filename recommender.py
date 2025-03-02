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
from database import get_db_pool
from config import RECOMMEND_CONFIG, TFIDF_PARAMS, BEHAVIOR_WEIGHTS, EXPOSURE_CONFIG, DATABASE_CONFIG, HOT_TOPICS_CONFIG
from cache import memory_cache, disk_cache, cached_get, cache_result, TimerContext
from monitor import track_performance, performance_collector

# 初始化数据库连接池
db_pool = get_db_pool()

# 如果BEHAVIOR_WEIGHTS中没有评论相关的行为权重，在这里添加默认值
if 'comment' not in BEHAVIOR_WEIGHTS:
    BEHAVIOR_WEIGHTS['comment'] = 4  # 评论行为权重设为4，比点赞高
if 'comment_like' not in BEHAVIOR_WEIGHTS:
    BEHAVIOR_WEIGHTS['comment_like'] = 2  # 评论点赞权重设为2，与点赞相同

# 并行处理的工作线程数
MAX_WORKERS = DATABASE_CONFIG.get('max_workers', 4)

class Recommender:
    def __init__(self, feature_processor=None):
        self.fp = feature_processor if feature_processor else FeatureProcessor()
        # 确保加载数据
        if not hasattr(self.fp, 'feature_names') or self.fp.feature_names is None:
            self.fp.load_data()
        
        # 记录最后一次热点生成时间
        self.last_hot_topics_time = None
        # 最近一次热点结果缓存
        self.hot_topics_cache = None
    
    @track_performance('hybrid_recommend', performance_collector)
    def hybrid_recommend(self, user_id, save_to_db=True):
        """混合推荐主逻辑"""
        start_time = time.time()
        
        # 先检查缓存中是否有此用户的推荐结果
        cache_key = f"recommend_result_{user_id}"
        cached_result = memory_cache.get(cache_key)
        if cached_result:
            performance_collector.increment_counter('cache_hits')
            if save_to_db and not self._check_recommendation_exists(user_id):
                self.save_recommendations_to_db(user_id, cached_result, 'hybrid_cached')
            return cached_result
        
        # 获取用户画像
        with TimerContext('get_user_profile'):
            user_profile = self.fp.get_user_profile(user_id)
        
        # 冷启动策略
        if user_profile is None:
            with TimerContext('popularity_recommend'):
                result = self.popularity_based_recommend()
                source = 'popularity'
        else:
            # 计算内容相似度
            with TimerContext('content_similarity'):
                tfidf_matrix = self.fp.get_tfidf_matrix()
                similarity = cosine_similarity([user_profile], tfidf_matrix)[0]
            
            # 排除已浏览帖子
            with TimerContext('exclude_viewed'):
                # 从user_views表获取用户已浏览的帖子
                sql = """
                SELECT DISTINCT post_id FROM user_views 
                WHERE user_id = %s AND post_id IS NOT NULL
                """
                viewed_posts = {row['post_id'] for row in db_pool.query(sql, (user_id,)) if row['post_id']}
                valid_mask = ~self.fp.posts['post_id'].isin(viewed_posts)
            
            # 获取可推送的曝光池帖子
            with TimerContext('get_exposure_posts'):
                exposure_posts = self.get_posts_from_exposure_pools(user_id)
            
            # 生成推荐结果
            with TimerContext('generate_recommendations'):
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    top_indices = similarity[valid_indices].argsort()[::-1][:RECOMMEND_CONFIG['top_n']]
                    recommended_posts = self.fp.posts.iloc[valid_indices[top_indices]]['post_id'].tolist()
                else:
                    recommended_posts = []
            
            # 合并推荐结果和曝光池结果
            with TimerContext('merge_recommendations'):
                result = self.merge_recommendations(recommended_posts, exposure_posts)
                source = 'hybrid'
        
        # 记录推送历史
        with TimerContext('record_exposure'):
            self.record_exposure(user_id, result)
        
        # 保存推荐结果到数据库
        if save_to_db:
            with TimerContext('save_to_db'):
                self.save_recommendations_to_db(user_id, result, source)
        
        # 缓存结果
        memory_cache.set(cache_key, result, 3600)  # 1小时过期
        
        # 记录处理时间
        process_time = int((time.time() - start_time) * 1000)
        performance_collector.record_metric('response_times', {
            'name': 'hybrid_recommend',
            'time_ms': process_time
        })
        
        return result
    
    def _check_recommendation_exists(self, user_id):
        """检查是否已有未过期的推荐结果"""
        sql = """
        SELECT id FROM user_recommendations 
        WHERE user_id = %s AND expire_time > NOW() AND is_read = 0
        LIMIT 1
        """
        result = db_pool.query(sql, (user_id,))
        return len(result) > 0
    
    @track_performance('save_recommendations', performance_collector)
    def save_recommendations_to_db(self, user_id, post_ids, source):
        """将推荐结果保存到数据库"""
        # 计算过期时间（默认24小时后）
        current_time = datetime.now()
        expire_time = current_time + timedelta(hours=DATABASE_CONFIG['recommendation_expire_hours'])
        
        # 将帖子ID列表转为JSON字符串
        post_ids_json = json.dumps(post_ids)
        
        # 保存到推荐结果表
        sql = """
        INSERT INTO user_recommendations 
        (user_id, recommendation_time, post_ids, source, expire_time)
        VALUES (%s, %s, %s, %s, %s)
        """
        params = (user_id, current_time, post_ids_json, source, expire_time)
        db_pool.execute(sql, params)
        
        # 记录推荐生成日志
        log_sql = """
        INSERT INTO recommend_logs 
        (user_id, recommend_time, post_count, process_time)
        VALUES (%s, %s, %s, %s)
        """
        process_time = 0  # 这里可以添加处理时间计算
        db_pool.execute(log_sql, (user_id, current_time, len(post_ids), process_time))
        
        # 更新计数器
        performance_collector.increment_counter('total_recommendations')
        
        return True
    
    @track_performance('batch_recommendations', performance_collector)
    def batch_generate_recommendations(self, user_count=100):
        """批量为活跃用户生成推荐结果，使用多线程并行处理"""
        # 查询最近活跃的用户 - 从多个行为表中聚合活跃用户
        sql = """
        SELECT user_id, MAX(activity_time) as last_active FROM (
            SELECT user_id, created_at as activity_time FROM post_likes 
            WHERE created_at > DATE_SUB(NOW(), INTERVAL %s DAY)
            UNION ALL
            SELECT user_id, created_at as activity_time FROM post_collects 
            WHERE created_at > DATE_SUB(NOW(), INTERVAL %s DAY)
            UNION ALL
            SELECT user_id, created_at as activity_time FROM user_views 
            WHERE created_at > DATE_SUB(NOW(), INTERVAL %s DAY)
            UNION ALL
            SELECT user_id, created_at as activity_time FROM user_search_records 
            WHERE created_at > DATE_SUB(NOW(), INTERVAL %s DAY)
            UNION ALL
            SELECT user_id, created_at as activity_time FROM comments
            WHERE created_at > DATE_SUB(NOW(), INTERVAL %s DAY)
            LIMIT %s
        ) as user_activities
        GROUP BY user_id
        ORDER BY last_active DESC
        LIMIT %s
        """
        days = DATABASE_CONFIG['active_user_days']
        active_users = db_pool.query(sql, (days, days, days, days, days, user_count))
        
        success_count = 0
        error_count = 0
        
        # 预加载TF-IDF矩阵和帖子数据（一次性加载，避免重复计算）
        self.fp.get_tfidf_matrix()
        
        # 创建线程锁，用于更新统计数据时防止竞争条件
        stats_lock = threading.Lock()
        
        # 用于存储结果的列表
        results = []
        
        def process_user(row):
            """处理单个用户的推荐，作为线程任务"""
            nonlocal success_count, error_count
            
            try:
                user_id = row['user_id'] if isinstance(row, dict) else row[0]
                
                # 检查是否已有推荐结果
                if self._check_recommendation_exists(user_id):
                    return
                
                # 生成推荐
                self.hybrid_recommend(user_id, save_to_db=True)
                
                # 更新统计数据
                with stats_lock:
                    success_count += 1
                    
            except Exception as e:
                with stats_lock:
                    error_count += 1
                print(f"[{datetime.now()}] 为用户{user_id}生成推荐失败: {str(e)}")
                performance_collector.increment_counter('failed_recommendations')
                
            return
        
        # 使用线程池并行处理用户推荐
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务
            futures = [executor.submit(process_user, row) for row in active_users]
            
            # 等待所有任务完成并显示进度
            completed = 0
            for future in futures:
                future.result()  # 等待任务完成
                completed += 1
                if completed % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / max(completed, 1)
                    print(f"[{datetime.now()}] 批量推荐进度: {completed}/{len(active_users)}, "
                          f"平均耗时: {avg_time:.2f}秒/用户")
        
        # 记录批量处理的性能指标
        total_time = time.time() - start_time
        if success_count > 0:
            avg_time = total_time / success_count
            performance_collector.record_metric('batch_processing', {
                'total_users': len(active_users),
                'success': success_count,
                'errors': error_count,
                'total_time': total_time,
                'avg_time': avg_time,
                'parallelism': MAX_WORKERS
            })
            
        return success_count
    
    @cache_result(memory_cache, key_prefix='popularity', expire_seconds=1800)
    def popularity_based_recommend(self):
        """热门帖子推荐 - 基于点赞、收藏和浏览的统计"""
        # 计算一个加权的热门帖子列表，使用热度分数而不是实时计算
        sql = """
        SELECT post_id
        FROM posts 
        WHERE created_at > DATE_SUB(NOW(), INTERVAL %s DAY)
        ORDER BY heat_score DESC
        LIMIT %s
        """
        params = (RECOMMEND_CONFIG['hot_days'], RECOMMEND_CONFIG['top_n'])
        return [row['post_id'] for row in db_pool.query(sql, params)]
    
    def get_posts_from_exposure_pools(self, user_id):
        """从曝光池中获取帖子"""
        # 获取曝光池配置
        global_ratio = EXPOSURE_CONFIG['global_ratio']
        pools = EXPOSURE_CONFIG['pools']
        
        # 计算需要获取的曝光池帖子数量
        total_posts = RECOMMEND_CONFIG['top_n']
        exposure_count = int(total_posts * global_ratio)
        
        # 查询用户已曝光过的帖子
        sql = """
        SELECT DISTINCT post_id 
        FROM post_exposures 
        WHERE user_id = %s AND exposure_time > DATE_SUB(NOW(), INTERVAL 7 DAY)
        """
        exposed_posts = {row['post_id'] for row in db_pool.query(sql, (user_id,))}
        
        # 从各曝光池获取帖子
        result = []
        
        for pool_level, config in sorted(pools.items()):
            # 计算此曝光池要获取的帖子数量
            pool_count = int(exposure_count * config['ratio'])
            if pool_count == 0:
                continue
                
            # 查询符合条件的帖子
            sql = """
            SELECT post_id 
            FROM posts 
            WHERE exposure_pool = %s
              AND exposure_count < %s
              AND created_at > DATE_SUB(NOW(), INTERVAL %s DAY)
              AND post_id NOT IN (
                  SELECT post_id FROM user_views 
                  WHERE user_id = %s AND post_id IS NOT NULL AND created_at > DATE_SUB(NOW(), INTERVAL 3 DAY)
              )
            ORDER BY RAND() 
            LIMIT %s
            """
            
            params = (
                pool_level, 
                config['max_exposures'],
                config['max_age_days'],
                user_id,
                pool_count
            )
            
            pool_posts = [row['post_id'] for row in db_pool.query(sql, params)]
            result.extend(pool_posts)
        
        # 如果所有曝光池获取的帖子不足，补充随机热门帖子
        if len(result) < exposure_count:
            needed = exposure_count - len(result)
            sql = """
            SELECT post_id 
            FROM posts 
            WHERE post_id NOT IN (%s)
              AND heat_score > 0
            ORDER BY heat_score DESC, RAND()
            LIMIT %s
            """
            
            # 避免SQL注入：构建安全的占位符
            excluded_ids = list(set(result) | exposed_posts)
            if excluded_ids:
                placeholders = ', '.join(['%s'] * len(excluded_ids))
                formatted_sql = sql.replace('(%s)', f'({placeholders})')
                random_posts = [row['post_id'] for row in db_pool.query(formatted_sql, excluded_ids + [needed])]
            else:
                # 没有需要排除的帖子
                random_posts = [row['post_id'] for row in db_pool.query(
                    "SELECT post_id FROM posts WHERE heat_score > 0 ORDER BY heat_score DESC, RAND() LIMIT %s", 
                    (needed,)
                )]
                
            result.extend(random_posts)
        
        # 随机打乱顺序
        random.shuffle(result)
        return result
    
    def merge_recommendations(self, content_posts, exposure_posts):
        """合并内容推荐和曝光推荐"""
        # 去重
        content_set = set(content_posts)
        exposure_set = set(exposure_posts)
        
        # 保留内容推荐中未在曝光推荐中的部分
        unique_content = list(content_set - exposure_set)
        
        # 确定最终推荐数量
        total_needed = RECOMMEND_CONFIG['top_n']
        exposure_ratio = EXPOSURE_CONFIG['global_ratio']
        
        # 计算最终各部分需要的数量
        exposure_count = min(len(exposure_posts), int(total_needed * exposure_ratio))
        content_count = min(len(unique_content), total_needed - exposure_count)
        
        # 选取最终结果
        final_content = unique_content[:content_count]
        final_exposure = exposure_posts[:exposure_count]
        
        # 合并结果
        result = final_content + final_exposure
        
        # 如果结果不足，补充热门帖子
        if len(result) < total_needed:
            needed = total_needed - len(result)
            excluded = set(result)
            
            # 查询热门帖子
            sql = """
            SELECT post_id 
            FROM posts 
            WHERE post_id NOT IN (%s)
            ORDER BY (view_count + like_count * 3 + collect_count * 5) DESC 
            LIMIT %s
            """
            
            # 构建安全的占位符
            if excluded:
                placeholders = ', '.join(['%s'] * len(excluded))
                formatted_sql = sql.replace('(%s)', f'({placeholders})')
                hot_posts = [row['post_id'] for row in db_pool.query(formatted_sql, list(excluded) + [needed])]
            else:
                hot_posts = [row['post_id'] for row in db_pool.query(
                    "SELECT post_id FROM posts ORDER BY (view_count + like_count * 3 + collect_count * 5) DESC LIMIT %s", 
                    (needed,)
                )]
                
            result.extend(hot_posts)
        
        # 随机打乱顺序
        random.shuffle(result)
        
        # 结果截断
        return result[:total_needed]
    
    def record_exposure(self, user_id, post_ids):
        """记录帖子曝光历史"""
        if not post_ids:
            return
            
        # 插入曝光记录
        now = datetime.now()
        values = [(user_id, post_id, now) for post_id in post_ids]
        
        sql = """
        INSERT INTO post_exposures (user_id, post_id, exposure_time)
        VALUES (%s, %s, %s)
        """
        db_pool.executemany(sql, values)
        
        # 批量更新帖子曝光次数 - 使用单个SQL语句
        post_ids_str = ', '.join([str(pid) for pid in post_ids])
        update_sql = f"""
        UPDATE posts 
        SET exposure_count = exposure_count + 1
        WHERE post_id IN ({post_ids_str})
        """
        db_pool.execute(update_sql)
    
    @track_performance('generate_hot_topics', performance_collector)
    def generate_hot_topics(self, count=50):
        """生成热点话题，按照算法计算帖子热度"""
        # 检查是否在缓存期内
        current_time = datetime.now()
        if (self.last_hot_topics_time and 
            (current_time - self.last_hot_topics_time).total_seconds() < 300 and
            self.hot_topics_cache):
            return self.hot_topics_cache
        
        # 计算帖子热度分数
        try:
            with TimerContext('calculate_heat_score'):
                # 预先计算评论点赞数，避免热度计算中的子查询
                comment_likes_sql = """
                SELECT c.post_id, COALESCE(SUM(cl.like_count), 0) as total_comment_likes
                FROM comments c 
                JOIN comment_likes cl ON c.comment_id = cl.comment_id 
                WHERE c.created_at > DATE_SUB(NOW(), INTERVAL 30 DAY)
                GROUP BY c.post_id
                """
                comment_likes = db_pool.query(comment_likes_sql)
                comment_likes_dict = {row['post_id']: row['total_comment_likes'] for row in comment_likes}
                
                # 更新热度分数 - 使用临时表/连接方式，避免子查询
                score_sql = """
                UPDATE posts p
                SET heat_score = (
                    COALESCE(view_count, 0) + 
                    COALESCE(like_count, 0) * 3 + 
                    COALESCE(collect_count, 0) * 5 + 
                    COALESCE(comment_count, 0) * 2
                )
                WHERE created_at > DATE_SUB(NOW(), INTERVAL 30 DAY)
                """
                db_pool.execute(score_sql)
                
                # 批量更新评论点赞热度
                if comment_likes:
                    # 准备批量更新参数
                    update_values = []
                    for post_id, like_count in comment_likes_dict.items():
                        update_values.append((like_count * 1.5, post_id))
                    
                    # 批量更新评论点赞热度
                    batch_sql = """
                    UPDATE posts 
                    SET heat_score = heat_score + %s 
                    WHERE post_id = %s
                    """
                    db_pool.executemany(batch_sql, update_values)
                
                # 衰减过老帖子的热度
                decay_sql = """
                UPDATE posts
                SET heat_score = heat_score * 0.8
                WHERE created_at < DATE_SUB(NOW(), INTERVAL 3 DAY)
                """
                db_pool.execute(decay_sql)
            
            # 根据热度更新帖子所在的曝光池
            with TimerContext('update_exposure_pools'):
                for pool_level, config in EXPOSURE_CONFIG['pools'].items():
                    threshold = config['heat_threshold']
                    max_age = config['max_age_days']
                    
                    sql = """
                    UPDATE posts 
                    SET exposure_pool = %s
                    WHERE heat_score >= %s
                      AND created_at > DATE_SUB(NOW(), INTERVAL %s DAY)
                      AND (exposure_pool < %s OR exposure_pool IS NULL)
                    """
                    db_pool.execute(sql, (pool_level, threshold, max_age, pool_level))
            
            # 获取前N个热点帖子
            with TimerContext('get_hot_topics'):
                sql = """
                SELECT post_id, post_title as title, heat_score
                FROM posts
                WHERE heat_score > %s
                ORDER BY heat_score DESC
                LIMIT %s
                """
                hot_topics = db_pool.query(sql, (HOT_TOPICS_CONFIG['min_heat_score'], count))
            
            # 记录热点生成历史
            with TimerContext('save_hot_topics_history'):
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                topics_json = json.dumps([{
                    'post_id': row['post_id'], 
                    'title': row['title'], 
                    'heat_score': row['heat_score']
                } for row in hot_topics])
                
                history_sql = """
                INSERT INTO hot_topics_history (generation_time, topics_json)
                VALUES (%s, %s)
                """
                db_pool.execute(history_sql, (timestamp, topics_json))
                
                # 更新当前热点表
                self.update_current_hot_topics(hot_topics)
            
            # 更新缓存时间和结果
            self.last_hot_topics_time = current_time
            self.hot_topics_cache = hot_topics
            
            return hot_topics
            
        except Exception as e:
            print(f"[{datetime.now()}] 生成热点话题失败: {str(e)}")
            performance_collector.increment_counter('failed_hot_topics')
            # 如果有缓存结果，返回缓存
            if self.hot_topics_cache:
                return self.hot_topics_cache
            return []
    
    def update_current_hot_topics(self, hot_topics):
        """更新当前热点表"""
        if not hot_topics:
            return
            
        # 清空当前热点表
        sql = "TRUNCATE TABLE current_hot_topics"
        db_pool.execute(sql)
        
        # 准备插入数据
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        values = []
        
        for i, row in enumerate(hot_topics):
            post_id = row['post_id'] if isinstance(row, dict) else row[0]
            title = row['title'] if isinstance(row, dict) else row[1]
            heat_score = row['heat_score'] if isinstance(row, dict) else row[2]
            values.append((current_time, post_id, title, heat_score, i+1))
        
        # 批量插入
        sql = """
        INSERT INTO current_hot_topics 
        (update_time, post_id, title, heat_score, rank_position)
        VALUES (%s, %s, %s, %s, %s)
        """
        db_pool.executemany(sql, values)
        
    def get_similar_posts(self, post_id, limit=20):
        """获取相似帖子推荐"""
        # 获取帖子特征
        post_sql = """
        SELECT p.hashtags, 
               (SELECT GROUP_CONCAT(content SEPARATOR ' ') 
                FROM comments 
                WHERE post_id = %s) as content
        FROM posts p 
        WHERE p.post_id = %s
        """
        post_result = db_pool.query(post_sql, (post_id, post_id))
        
        if not post_result:
            return []
        
        post_tags = post_result[0]['hashtags'] or ''
        post_comments = post_result[0]['content'] or ''
        
        # 组合成特征文本
        feature_text = post_tags.replace(',', ' ') + ' ' + post_comments
        if not feature_text.strip():
            return self.popularity_based_recommend()[:limit]
        
        # 从缓存获取相似帖子结果
        cache_key = f"similar_posts_{post_id}_{limit}"
        cached_result = memory_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # 转换特征为向量
        post_vec = self.fp.tfidf.transform([feature_text]).toarray()[0]
        
        # 计算与所有帖子的相似度
        tfidf_matrix = self.fp.get_tfidf_matrix()
        similarities = cosine_similarity([post_vec], tfidf_matrix)[0]
        
        # 获取相似帖子索引（排除自身）
        post_indices = list(range(len(self.fp.posts)))
        target_idx = self.fp.posts[self.fp.posts['post_id'] == post_id].index
        if len(target_idx) > 0:
            post_indices.remove(target_idx[0])
        
        # 获取最相似的帖子
        similar_indices = sorted([(i, similarities[i]) for i in post_indices], 
                                key=lambda x: x[1], reverse=True)[:limit]
        
        # 返回帖子ID列表
        similar_posts = [self.fp.posts.iloc[idx[0]]['post_id'] for idx in similar_indices]
        
        # 缓存结果（3小时过期）
        memory_cache.set(cache_key, similar_posts, 10800)
        
        return similar_posts
        
    def refresh_model(self):
        """刷新推荐模型"""
        self.fp.fit_model()
        # 清空缓存
        memory_cache.clear()
        return True

class FeatureProcessor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(**TFIDF_PARAMS)
        self.feature_names = None
        self.posts = None
        self.tfidf_matrix = None
        self.svd_model = None
        self.svd_features = None
    
    def load_data(self):
        """加载帖子数据"""
        # 先尝试从缓存读取
        cached_posts = disk_cache.get('posts_data')
        if cached_posts:
            self.posts = cached_posts
        else:
            # 从posts表加载帖子数据，使用hashtags作为标签，同时加载评论内容摘要
            sql = """
            SELECT p.post_id, p.post_title as title, p.hashtags as tags,
                   COALESCE(GROUP_CONCAT(DISTINCT c.content SEPARATOR ' '), '') as content
            FROM posts p
            LEFT JOIN comments c ON p.post_id = c.post_id
            GROUP BY p.post_id, p.post_title, p.hashtags
            """
            results = db_pool.query(sql)
            
            # 转换结果为DataFrame
            self.posts = pd.DataFrame(results)
            self.posts['tags'] = self.posts['tags'].fillna('').str.replace(',', ' ')
            
            # 合并标签和评论内容作为特征（评论内容权重较低）
            self.posts['features'] = self.posts.apply(
                lambda x: x['tags'] + ' ' + ' '.join(x['content'].split()[:50]) 
                if x['content'] else x['tags'], axis=1
            )
            
            # 缓存结果
            disk_cache.set('posts_data', self.posts)
        
        # 如果已有模型就加载，否则训练
        try:
            self.tfidf = joblib.load('tfidf_model.pkl')
            self.feature_names = self.tfidf.get_feature_names_out()
            
            # 尝试加载SVD模型
            try:
                self.svd_model = joblib.load('svd_model.pkl')
            except:
                pass
                
        except:
            self.fit_model()
    
    def fit_model(self):
        """训练TF-IDF模型"""
        if self.posts is None or len(self.posts) == 0:
            self.load_data()
        
        if len(self.posts) > 0:
            # 训练TF-IDF
            feature_texts = self.posts['features'].fillna('')
            self.tfidf = TfidfVectorizer(**TFIDF_PARAMS)
            self.tfidf_matrix = self.tfidf.fit_transform(feature_texts)
            self.feature_names = self.tfidf.get_feature_names_out()
            
            # 保存模型
            joblib.dump(self.tfidf, 'tfidf_model.pkl')
            
            # 训练SVD降维
            if self.tfidf_matrix.shape[1] > 100:
                n_components = min(100, self.tfidf_matrix.shape[1] - 1)
                self.svd_model = TruncatedSVD(n_components=n_components)
                self.svd_features = self.svd_model.fit_transform(self.tfidf_matrix)
                joblib.dump(self.svd_model, 'svd_model.pkl')
    
    @track_performance('get_tfidf_matrix', performance_collector)
    def get_tfidf_matrix(self):
        """获取TF-IDF矩阵（带缓存）"""
        if self.tfidf_matrix is not None:
            return self.tfidf_matrix
            
        if self.posts is None:
            self.load_data()
            
        # 计算TF-IDF矩阵
        feature_texts = self.posts['features'].fillna('')
        self.tfidf_matrix = self.tfidf.transform(feature_texts)
        return self.tfidf_matrix
    
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
        
        days = RECOMMEND_CONFIG['recent_days']
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
            
            weight = BEHAVIOR_WEIGHTS.get(behavior, 1)
            try:
                # 检查缓存中是否已有此标签的向量
                if tags not in tag_vectors:
                    tag_vectors[tags] = self.tfidf.transform([tags]).toarray()[0]
                
                user_vector += tag_vectors[tags] * weight
            except Exception as e:
                print(f"处理用户{user_id}的兴趣向量失败: {str(e)}")
                continue
        
        # 归一化
        norm = np.linalg.norm(user_vector)
        if norm == 0:
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
        days = RECOMMEND_CONFIG['recent_days']
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
                # 计算余弦相似度
                similarity = np.dot(user_profile, other_profile) / (
                    np.linalg.norm(user_profile) * np.linalg.norm(other_profile)
                )
                similarities.append((other_id, similarity))
        
        # 排序返回最相似的用户
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [user[0] for user in similarities[:limit]]