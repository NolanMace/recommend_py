import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import pandas as pd
from typing import List, Dict, Any
import pickle
import os
from datetime import datetime, timedelta

class BaseRecommender:
    """推荐算法基类"""
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"recommender.{name}")
    
    def fit(self, data: Dict[str, Any]):
        """训练模型"""
        raise NotImplementedError
    
    def recommend(self, user_id: int, top_n: int = 10) -> List[Dict]:
        """为用户生成推荐"""
        raise NotImplementedError
    
    def save_model(self, path: str):
        """保存模型"""
        raise NotImplementedError
    
    def load_model(self, path: str):
        """加载模型"""
        raise NotImplementedError

class TFIDFRecommender(BaseRecommender):
    """基于TF-IDF的内容推荐"""
    def __init__(self):
        super().__init__("tfidf")
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.post_vectors = None
        self.posts_df = None
        self.post_id_to_idx = {}
        self.idx_to_post_id = {}
    
    def fit(self, data: Dict[str, Any]):
        self.posts_df = data['posts']
        
        # 合并标题和内容作为文本特征
        text_features = self.posts_df['title'] + " " + self.posts_df['content']
        self.post_vectors = self.vectorizer.fit_transform(text_features)
        
        # 建立ID映射
        for idx, post_id in enumerate(self.posts_df['post_id']):
            self.post_id_to_idx[post_id] = idx
            self.idx_to_post_id[idx] = post_id
            
        self.logger.info(f"TF-IDF模型训练完成，文档数量: {len(self.posts_df)}")
        return self
    
    def recommend(self, user_data: Dict, top_n: int = 10) -> List[Dict]:
        """基于用户历史交互帖子和兴趣标签推荐内容"""
        user_id = user_data['user_id']
        liked_posts = user_data.get('liked_posts', [])
        viewed_posts = user_data.get('viewed_posts', [])
        user_tags = user_data.get('interest_tags', [])
        
        # 如果用户没有历史行为，返回热门帖子
        if not liked_posts and not viewed_posts and not user_tags:
            return self._get_popular_posts(top_n)
        
        # 基于用户历史行为构建用户向量
        user_vector = self._build_user_vector(liked_posts, viewed_posts, user_tags)
        
        # 计算用户向量与所有帖子的相似度
        similarities = cosine_similarity(user_vector, self.post_vectors).flatten()
        
        # 获取已经看过的帖子索引
        viewed_indices = [self.post_id_to_idx[pid] for pid in viewed_posts 
                         if pid in self.post_id_to_idx]
        
        # 设置已看过的帖子相似度为-1，确保不会被推荐
        for idx in viewed_indices:
            similarities[idx] = -1
            
        # 获取相似度最高的N个帖子索引
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        # 构建推荐结果
        recommendations = []
        for idx in top_indices:
            post_id = self.idx_to_post_id[idx]
            post_info = self.posts_df[self.posts_df['post_id'] == post_id].iloc[0].to_dict()
            recommendations.append({
                'post_id': post_id,
                'score': float(similarities[idx]),
                'title': post_info['title'],
                'algorithm': 'tfidf'
            })
            
        return recommendations
    
    def _build_user_vector(self, liked_posts, viewed_posts, user_tags):
        """构建用户兴趣向量"""
        # 权重设置
        like_weight = 3.0
        view_weight = 1.0
        tag_weight = 2.0
        
        # 初始化用户向量
        user_vector = np.zeros((1, self.post_vectors.shape[1]))
        
        # 添加点赞帖子的向量
        for post_id in liked_posts:
            if post_id in self.post_id_to_idx:
                idx = self.post_id_to_idx[post_id]
                user_vector += like_weight * self.post_vectors[idx].toarray()
        
        # 添加浏览帖子的向量
        for post_id in viewed_posts:
            if post_id in self.post_id_to_idx:
                idx = self.post_id_to_idx[post_id]
                user_vector += view_weight * self.post_vectors[idx].toarray()
        
        # 如果有标签数据，也添加到用户向量中
        if user_tags:
            tag_texts = " ".join(user_tags)
            tag_vector = self.vectorizer.transform([tag_texts])
            user_vector += tag_weight * tag_vector.toarray()
            
        # 归一化用户向量
        vec_norm = np.linalg.norm(user_vector)
        if vec_norm > 0:
            user_vector = user_vector / vec_norm
            
        return user_vector
        
    def _get_popular_posts(self, top_n):
        """获取热门帖子（用于冷启动）"""
        # 这里假设posts_df有一个popularity字段表示热度
        popular_posts = self.posts_df.sort_values('heat_score', ascending=False).head(top_n)
        
        recommendations = []
        for _, post in popular_posts.iterrows():
            recommendations.append({
                'post_id': post['post_id'],
                'score': float(0.5),  # 默认分数
                'title': post['title'],
                'algorithm': 'popularity'
            })
            
        return recommendations
    
    def save_model(self, path: str):
        """保存模型到文件"""
        model_data = {
            'vectorizer': self.vectorizer,
            'post_vectors': self.post_vectors,
            'post_id_to_idx': self.post_id_to_idx,
            'idx_to_post_id': self.idx_to_post_id,
            'timestamp': datetime.now()
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"模型已保存到: {path}")
        
    def load_model(self, path: str):
        """从文件加载模型"""
        if not os.path.exists(path):
            self.logger.warning(f"模型文件不存在: {path}")
            return False
            
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.vectorizer = model_data['vectorizer']
        self.post_vectors = model_data['post_vectors']
        self.post_id_to_idx = model_data['post_id_to_idx']
        self.idx_to_post_id = model_data['idx_to_post_id']
        
        self.logger.info(f"模型已加载，时间戳: {model_data['timestamp']}")
        return True

class CollaborativeFilteringRecommender(BaseRecommender):
    """基于协同过滤的推荐"""
    def __init__(self):
        super().__init__("collaborative")
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.users_df = None
        self.posts_df = None
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        self.post_id_to_idx = {}
        self.idx_to_post_id = {}
        
    def fit(self, data: Dict[str, Any]):
        """训练协同过滤模型"""
        self.users_df = data['users']
        self.posts_df = data['posts']
        interactions_df = data['interactions']
        
        # 建立用户和帖子的ID映射
        for idx, user_id in enumerate(self.users_df['user_id'].unique()):
            self.user_id_to_idx[user_id] = idx
            self.idx_to_user_id[idx] = user_id
            
        for idx, post_id in enumerate(self.posts_df['post_id'].unique()):
            self.post_id_to_idx[post_id] = idx
            self.idx_to_post_id[idx] = post_id
        
        # 构建用户-物品交互矩阵
        n_users = len(self.user_id_to_idx)
        n_items = len(self.post_id_to_idx)
        self.user_item_matrix = np.zeros((n_users, n_items))
        
        # 填充交互矩阵
        for _, row in interactions_df.iterrows():
            user_id = row['user_id']
            post_id = row['post_id']
            interaction_type = row['interaction_type']
            
            if user_id in self.user_id_to_idx and post_id in self.post_id_to_idx:
                u_idx = self.user_id_to_idx[user_id]
                i_idx = self.post_id_to_idx[post_id]
                
                # 根据不同的交互类型赋予不同的权重
                if interaction_type == 'view':
                    weight = 1.0
                elif interaction_type == 'like':
                    weight = 3.0
                elif interaction_type == 'collect':
                    weight = 4.0
                elif interaction_type == 'comment':
                    weight = 5.0
                else:
                    weight = 1.0
                    
                self.user_item_matrix[u_idx, i_idx] = weight
        
        # 计算物品相似度矩阵（基于物品的协同过滤）
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        
        self.logger.info(f"协同过滤模型训练完成，用户数: {n_users}, 物品数: {n_items}")
        return self
        
    def recommend(self, user_data: Dict, top_n: int = 10) -> List[Dict]:
        """为用户生成推荐"""
        user_id = user_data['user_id']
        viewed_posts = user_data.get('viewed_posts', [])
        
        # 如果是新用户，返回热门帖子
        if user_id not in self.user_id_to_idx:
            return self._get_popular_posts(top_n)
            
        u_idx = self.user_id_to_idx[user_id]
        user_ratings = self.user_item_matrix[u_idx]
        
        # 已评分的物品索引
        rated_indices = np.where(user_ratings > 0)[0]
        
        # 如果用户没有任何交互，返回热门帖子
        if len(rated_indices) == 0:
            return self._get_popular_posts(top_n)
            
        # 计算预测评分
        predictions = np.zeros(len(self.post_id_to_idx))
        for i_idx in range(len(self.post_id_to_idx)):
            # 跳过用户已交互的帖子
            if user_ratings[i_idx] > 0:
                predictions[i_idx] = -1
                continue
                
            # 计算该物品与用户已交互物品的加权评分
            weighted_sum = 0
            similarity_sum = 0
            
            for rated_idx in rated_indices:
                similarity = self.item_similarity[i_idx, rated_idx]
                weighted_sum += similarity * user_ratings[rated_idx]
                similarity_sum += abs(similarity)
                
            predictions[i_idx] = weighted_sum / similarity_sum if similarity_sum > 0 else 0
        
        # 获取已经看过的帖子索引
        viewed_indices = [self.post_id_to_idx[pid] for pid in viewed_posts 
                         if pid in self.post_id_to_idx]
        
        # 设置已看过的帖子预测分数为-1，确保不会被推荐
        for idx in viewed_indices:
            predictions[idx] = -1
            
        # 获取预测分数最高的N个帖子索引
        top_indices = np.argsort(predictions)[::-1][:top_n]
        
        # 构建推荐结果
        recommendations = []
        for idx in top_indices:
            if predictions[idx] <= 0:
                continue
                
            post_id = self.idx_to_post_id[idx]
            post_info = self.posts_df[self.posts_df['post_id'] == post_id].iloc[0].to_dict()
            
            recommendations.append({
                'post_id': post_id,
                'score': float(predictions[idx]),
                'title': post_info['title'],
                'algorithm': 'collaborative'
            })
            
        return recommendations
    
    def _get_popular_posts(self, top_n):
        """获取热门帖子（用于冷启动）"""
        popular_posts = self.posts_df.sort_values('heat_score', ascending=False).head(top_n)
        
        recommendations = []
        for _, post in popular_posts.iterrows():
            recommendations.append({
                'post_id': post['post_id'],
                'score': float(0.5),  # 默认分数
                'title': post['title'],
                'algorithm': 'popularity'
            })
            
        return recommendations
        
    def save_model(self, path: str):
        """保存模型到文件"""
        model_data = {
            'user_item_matrix': self.user_item_matrix,
            'item_similarity': self.item_similarity,
            'user_id_to_idx': self.user_id_to_idx,
            'idx_to_user_id': self.idx_to_user_id,
            'post_id_to_idx': self.post_id_to_idx,
            'idx_to_post_id': self.idx_to_post_id,
            'timestamp': datetime.now()
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"协同过滤模型已保存到: {path}")
        
    def load_model(self, path: str):
        """从文件加载模型"""
        if not os.path.exists(path):
            self.logger.warning(f"模型文件不存在: {path}")
            return False
            
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.user_item_matrix = model_data['user_item_matrix']
        self.item_similarity = model_data['item_similarity']
        self.user_id_to_idx = model_data['user_id_to_idx']
        self.idx_to_user_id = model_data['idx_to_user_id']
        self.post_id_to_idx = model_data['post_id_to_idx']
        self.idx_to_post_id = model_data['idx_to_post_id']
        
        self.logger.info(f"协同过滤模型已加载，时间戳: {model_data['timestamp']}")
        return True

class HybridRecommender(BaseRecommender):
    """混合推荐器，综合使用多种推荐算法"""
    
    def __init__(self):
        super().__init__("hybrid")
        # 初始化子推荐器
        self.tfidf_recommender = TFIDFRecommender()
        self.cf_recommender = CollaborativeFilteringRecommender()
        
        # 各推荐器权重
        self.weights = {
            'tfidf': 0.6,
            'cf': 0.4
        }
        
        # 相似度阈值，低于此值的不会被推荐
        self.similarity_threshold = 0.1
        
        # 辅助数据
        self.category_relations = {}  # 分类关系图
        self.posts_df = None  # 帖子数据
        self.posts_by_category = {}  # 按分类索引的帖子
        
    def fit(self, data: Dict[str, Any]):
        """训练所有子推荐器"""
        # 训练TF-IDF推荐器
        self.tfidf_recommender.fit(data)
        
        # 训练协同过滤推荐器
        self.cf_recommender.fit(data)
        
        # 保存帖子数据用于热门推荐和最新推荐
        self.posts_df = data['posts']
        
        # 构建分类关系图和分类索引
        self._build_category_relations(data)
        
        self.logger.info("混合推荐器训练完成")
        return self
    
    def set_similarity_threshold(self, threshold: float):
        """设置相似度阈值"""
        self.similarity_threshold = threshold
        self.logger.debug(f"相似度阈值已设置为: {threshold}")
        
    def _build_category_relations(self, data: Dict[str, Any]):
        """构建分类关系图和分类索引"""
        # 如果数据中有分类关系数据
        if 'category_relations' in data:
            self.category_relations = data['category_relations']
        else:
            # 否则从帖子数据中构建简单的分类索引
            self.posts_by_category = {}
            for _, post in self.posts_df.iterrows():
                if 'category' in post:
                    category = post['category']
                    if category not in self.posts_by_category:
                        self.posts_by_category[category] = []
                    self.posts_by_category[category].append(post['post_id'])
                    
            # 构建简单的分类关系图(基于共现)
            if 'interactions' in data:
                # 计算分类共现关系
                category_co_view = {}
                interactions_df = data['interactions']
                
                # 获取用户浏览历史中的分类集合
                user_categories = {}
                for _, row in interactions_df.iterrows():
                    user_id = row['user_id']
                    post_id = row['post_id']
                    
                    # 查找帖子所属分类
                    post_row = self.posts_df[self.posts_df['post_id'] == post_id]
                    if not post_row.empty and 'category' in post_row.iloc[0]:
                        category = post_row.iloc[0]['category']
                        
                        if user_id not in user_categories:
                            user_categories[user_id] = set()
                        user_categories[user_id].add(category)
                
                # 计算分类共现
                for user_id, categories in user_categories.items():
                    categories = list(categories)
                    for i in range(len(categories)):
                        for j in range(i+1, len(categories)):
                            cat1, cat2 = categories[i], categories[j]
                            pair = tuple(sorted([cat1, cat2]))
                            
                            if pair not in category_co_view:
                                category_co_view[pair] = 0
                            category_co_view[pair] += 1
                
                # 构建分类关系图
                for (cat1, cat2), count in category_co_view.items():
                    if count >= 3:  # 共现超过3次认为有关联
                        if cat1 not in self.category_relations:
                            self.category_relations[cat1] = []
                        if cat2 not in self.category_relations:
                            self.category_relations[cat2] = []
                            
                        self.category_relations[cat1].append(cat2)
                        self.category_relations[cat2].append(cat1)
    
    def expand_user_interests(self, user_data: Dict) -> Dict:
        """扩展用户兴趣标签，返回扩展后的用户数据"""
        if 'interest_tags' not in user_data or not self.category_relations:
            return user_data
            
        result = user_data.copy()
        original_tags = set(user_data['interest_tags'])
        expanded_tags = set(original_tags)
        
        # 为每个兴趣标签添加相关标签
        for tag in original_tags:
            if tag in self.category_relations:
                related_tags = self.category_relations[tag]
                # 限制只添加最相关的3个标签
                for related_tag in related_tags[:3]:
                    expanded_tags.add(related_tag)
        
        # 更新用户兴趣标签
        result['interest_tags'] = list(expanded_tags)
        
        # 记录扩展了多少标签
        added = len(expanded_tags) - len(original_tags)
        if added > 0:
            self.logger.debug(f"用户兴趣标签扩展: +{added}个标签")
            
        return result
    
    def recommend(self, user_data: Dict, top_n: int = 10) -> List[Dict]:
        """使用混合策略推荐内容"""
        # 如果用户没有任何历史行为，直接返回热门推荐
        if not user_data.get('liked_posts') and not user_data.get('viewed_posts') and not user_data.get('interest_tags'):
            return self.get_hot_recommendations(top_n)
            
        # 获取各推荐器的推荐结果
        tfidf_recs = self.tfidf_recommender.recommend(user_data, top_n=top_n*2)
        cf_recs = self.cf_recommender.recommend(user_data, top_n=top_n*2)
        
        # 合并结果
        all_recommendations = []
        
        # 添加TF-IDF推荐结果
        for rec in tfidf_recs:
            # 检查相似度是否高于阈值
            if rec.get('score', 0) >= self.similarity_threshold:
                all_recommendations.append(rec)
        
        # 添加协同过滤推荐结果
        cf_post_ids = set(r['post_id'] for r in all_recommendations)
        for rec in cf_recs:
            # 防止重复并检查相似度
            if rec['post_id'] not in cf_post_ids and rec.get('score', 0) >= self.similarity_threshold:
                all_recommendations.append(rec)
                cf_post_ids.add(rec['post_id'])
        
        # 按加权得分排序
        for rec in all_recommendations:
            algorithm = rec.get('algorithm', '')
            base_score = rec.get('score', 0)
            
            # 应用权重
            if algorithm == 'tfidf':
                rec['weighted_score'] = base_score * self.weights['tfidf']
            elif algorithm == 'collaborative':
                rec['weighted_score'] = base_score * self.weights['cf']
            else:
                rec['weighted_score'] = base_score * 0.5
        
        # 排序并截取top_n结果
        all_recommendations.sort(key=lambda x: x.get('weighted_score', 0), reverse=True)
        return all_recommendations[:top_n]
    
    def get_hot_recommendations(self, top_n: int = 10, exclude_ids: List = None) -> List[Dict]:
        """获取热门推荐结果"""
        if self.posts_df is None or self.posts_df.empty:
            return []
            
        # 默认排除列表
        if exclude_ids is None:
            exclude_ids = []
        
        # 创建排除集合以加速查找
        exclude_set = set(exclude_ids)
        
        try:
            # 按热度得分降序排序
            sorted_posts = self.posts_df.sort_values('heat_score', ascending=False)
            
            # 过滤出未被排除的帖子
            recommendations = []
            for _, post in sorted_posts.iterrows():
                if post['post_id'] not in exclude_set:
                    recommendations.append({
                        'post_id': post['post_id'],
                        'title': post['title'],
                        'score': float(post.get('heat_score', 0.5)),
                        'algorithm': 'hot'
                    })
                    
                    if len(recommendations) >= top_n:
                        break
            
            return recommendations
        except Exception as e:
            self.logger.error(f"获取热门推荐失败: {e}")
            return []
    
    def get_new_recommendations(self, top_n: int = 10, exclude_ids: List = None) -> List[Dict]:
        """获取最新推荐结果"""
        if self.posts_df is None or self.posts_df.empty:
            return []
            
        # 默认排除列表
        if exclude_ids is None:
            exclude_ids = []
        
        # 创建排除集合以加速查找
        exclude_set = set(exclude_ids)
        
        try:
            # 确保有发布时间字段
            if 'publish_time' not in self.posts_df.columns:
                return []
                
            # 按发布时间降序排序
            sorted_posts = self.posts_df.sort_values('publish_time', ascending=False)
            
            # 过滤出未被排除的帖子
            recommendations = []
            for _, post in sorted_posts.iterrows():
                if post['post_id'] not in exclude_set:
                    recommendations.append({
                        'post_id': post['post_id'],
                        'title': post['title'],
                        'score': 0.3,  # 默认分数
                        'algorithm': 'newest'
                    })
                    
                    if len(recommendations) >= top_n:
                        break
            
            return recommendations
        except Exception as e:
            self.logger.error(f"获取最新推荐失败: {e}")
            return []
    
    def save_model(self, path: str):
        """保存模型"""
        # 创建目录结构
        model_dir = os.path.dirname(path)
        os.makedirs(model_dir, exist_ok=True)
        
        # 构建混合模型数据
        hybrid_data = {
            'weights': self.weights,
            'similarity_threshold': self.similarity_threshold,
            'category_relations': self.category_relations,
            'timestamp': datetime.now()
        }
        
        # 保存混合模型数据
        with open(path, 'wb') as f:
            pickle.dump(hybrid_data, f)
            
        # 保存子模型
        tfidf_path = os.path.join(model_dir, 'tfidf_model.pkl')
        cf_path = os.path.join(model_dir, 'cf_model.pkl')
        
        self.tfidf_recommender.save_model(tfidf_path)
        self.cf_recommender.save_model(cf_path)
        
        self.logger.info(f"混合推荐模型已保存: {path}")
        
    def load_model(self, path: str) -> bool:
        """加载模型"""
        # 检查主模型文件
        if not os.path.exists(path):
            self.logger.warning(f"混合模型文件不存在: {path}")
            return False
            
        try:
            # 加载混合模型数据
            with open(path, 'rb') as f:
                hybrid_data = pickle.load(f)
                
            self.weights = hybrid_data.get('weights', self.weights)
            self.similarity_threshold = hybrid_data.get('similarity_threshold', self.similarity_threshold)
            self.category_relations = hybrid_data.get('category_relations', {})
            
            # 加载子模型
            model_dir = os.path.dirname(path)
            tfidf_path = os.path.join(model_dir, 'tfidf_model.pkl')
            cf_path = os.path.join(model_dir, 'cf_model.pkl')
            
            tfidf_loaded = self.tfidf_recommender.load_model(tfidf_path)
            cf_loaded = self.cf_recommender.load_model(cf_path)
            
            if tfidf_loaded and cf_loaded:
                self.logger.info(f"混合推荐模型加载成功，时间戳: {hybrid_data.get('timestamp')}")
                return True
            else:
                self.logger.warning("部分子模型加载失败")
                return False
                
        except Exception as e:
            self.logger.error(f"加载混合推荐模型失败: {e}")
            return False

class RecommendationEngine:
    """推荐引擎主类，提供推荐服务接口"""
    def __init__(self, model_dir='./models'):
        self.model_dir = model_dir
        self.recommender = HybridRecommender()
        self.logger = logging.getLogger("recommendation_engine")
        self.model_loaded = False
        self.model_path = os.path.join(model_dir, 'hybrid_model.pkl')
        
        # 尝试加载模型
        self.load_model()
        
    def train_model(self, data: Dict[str, Any]):
        """训练推荐模型"""
        self.logger.info("开始训练推荐模型...")
        start_time = datetime.now()
        
        self.recommender.fit(data)
        
        # 保存模型
        self.save_model()
        
        end_time = datetime.now()
        self.logger.info(f"模型训练完成，耗时: {(end_time - start_time).total_seconds()}秒")
        self.model_loaded = True
        
    def get_recommendations(self, user_data: Dict, top_n: int = 10, 
                           similarity_threshold: float = 0.1) -> List[Dict]:
        """获取用户推荐
        
        Args:
            user_data: 用户数据，包含user_id, viewed_posts等信息
            top_n: 推荐数量
            similarity_threshold: 相似度阈值，低于此值的不会被推荐
            
        Returns:
            推荐列表
        """
        if not self.model_loaded:
            self.logger.warning("模型未加载，尝试加载模型...")
            if not self.load_model():
                self.logger.error("模型加载失败，无法提供推荐")
                return []
        
        try:
            # 设置相似度阈值
            if hasattr(self.recommender, 'set_similarity_threshold'):
                self.recommender.set_similarity_threshold(similarity_threshold)
                
            recommendations = self.recommender.recommend(user_data, top_n)
            return recommendations
        except Exception as e:
            self.logger.error(f"生成推荐时出错: {e}")
            return []
    
    def get_diverse_recommendations(self, user_data: Dict, top_n: int = 10, 
                                  diversity_level: float = 0.3) -> List[Dict]:
        """获取多样化推荐，增加结果多样性
        
        Args:
            user_data: 用户数据
            top_n: 推荐数量
            diversity_level: 多样性程度，越高结果越多样
            
        Returns:
            推荐列表
        """
        if not self.model_loaded:
            if not self.load_model():
                return []
        
        try:
            # 获取更多候选项，然后进行多样性过滤
            candidates_count = int(top_n * (1 + diversity_level * 2))
            candidates = self.recommender.recommend(user_data, candidates_count)
            
            if not candidates:
                return []
                
            # 多样性过滤逻辑
            # 1. 按算法类型分组
            grouped = {}
            for rec in candidates:
                algo = rec.get('algorithm', 'unknown')
                if algo not in grouped:
                    grouped[algo] = []
                grouped[algo].append(rec)
                
            # 2. 从每个分组中抽取一定比例
            result = []
            remaining = top_n
            algorithms = list(grouped.keys())
            
            # 均匀分配各算法数量
            per_algo = max(1, remaining // len(algorithms))
            
            for algo in algorithms:
                # 每个算法取top结果
                items = grouped[algo][:per_algo]
                result.extend(items)
                remaining -= len(items)
                
            # 如果还有剩余容量，填充得分最高的其他项
            if remaining > 0:
                # 扁平化剩余候选项
                flat_remaining = []
                for algo in algorithms:
                    flat_remaining.extend(grouped[algo][per_algo:])
                
                # 排序并取top结果
                flat_remaining.sort(key=lambda x: x.get('score', 0), reverse=True)
                result.extend(flat_remaining[:remaining])
                
            return result[:top_n]
        except Exception as e:
            self.logger.error(f"生成多样化推荐时出错: {e}")
            return []
    
    def get_expanded_recommendations(self, user_data: Dict, top_n: int = 10,
                                   expansion_level: int = 1) -> List[Dict]:
        """获取扩展推荐，通过降低相似度门槛和扩大候选范围来处理推荐不足情况
        
        Args:
            user_data: 用户数据
            top_n: 推荐数量
            expansion_level: 扩展级别，越高扩展越多
            
        Returns:
            推荐列表
        """
        if not self.model_loaded:
            if not self.load_model():
                return []
        
        try:
            # 根据扩展级别设置相似度阈值
            base_threshold = 0.1
            adjusted_threshold = max(0.01, base_threshold - (expansion_level * 0.02))
            
            # 根据扩展级别增加获取数量
            expanded_count = int(top_n * (1 + expansion_level * 0.5))
            expanded_count = min(expanded_count, top_n * 3)  # 最多获取3倍
            
            # 使用新的相似度阈值获取推荐
            recommendations = self.get_recommendations(
                user_data, 
                top_n=expanded_count,
                similarity_threshold=adjusted_threshold
            )
            
            # 如果扩展获取后仍然不足
            if len(recommendations) < top_n:
                self.logger.info(f"扩展推荐后仍不足，尝试兴趣扩展: {len(recommendations)}/{top_n}")
                
                # 尝试扩展用户兴趣标签
                if 'interest_tags' in user_data and hasattr(self.recommender, 'expand_user_interests'):
                    expanded_user_data = user_data.copy()
                    # 调用推荐器的用户兴趣扩展方法
                    expanded_user_data = self.recommender.expand_user_interests(expanded_user_data)
                    
                    # 使用扩展后的用户数据再次获取推荐
                    more_recommendations = self.get_recommendations(
                        expanded_user_data,
                        top_n=top_n - len(recommendations),
                        similarity_threshold=adjusted_threshold
                    )
                    
                    # 添加来源标记
                    for rec in more_recommendations:
                        rec['source'] = 'interest_expansion'
                        
                    recommendations.extend(more_recommendations)
            
            return recommendations[:top_n]
        except Exception as e:
            self.logger.error(f"生成扩展推荐时出错: {e}")
            return []
    
    def get_fallback_recommendations(self, top_n: int = 10) -> List[Dict]:
        """获取兜底推荐，用于冷启动或推荐数量不足情况
        
        Args:
            top_n: 推荐数量
            
        Returns:
            推荐列表
        """
        try:
            # 首先尝试获取热门内容
            recommendations = []
            
            if hasattr(self.recommender, 'get_hot_recommendations'):
                hot_recs = self.recommender.get_hot_recommendations(top_n)
                recommendations.extend(hot_recs)
            
            # 如果热门内容不足，使用最新内容补充
            if len(recommendations) < top_n and hasattr(self.recommender, 'get_new_recommendations'):
                new_recs = self.recommender.get_new_recommendations(top_n - len(recommendations))
                recommendations.extend(new_recs)
                
            # 确保有结果返回
            if not recommendations:
                self.logger.warning(f"无法获取兜底推荐，返回空列表")
                
            return recommendations[:top_n]
        except Exception as e:
            self.logger.error(f"生成兜底推荐时出错: {e}")
            return []
    
    def get_user_recommendations(self, user_id, count=20, offset=0, 
                               similarity_threshold=None, diversity_weight=None,
                               excluded_items=None):
        """获取用户推荐，支持分页和无限滚动
        
        Args:
            user_id: 用户ID
            count: 需要的推荐数量
            offset: 偏移量，用于分页
            similarity_threshold: 相似度阈值，None则使用默认值
            diversity_weight: 多样性权重，None则使用默认值
            excluded_items: 需要排除的内容ID集合
            
        Returns:
            推荐列表
        """
        start_time = datetime.now()
        self.logger.info(f"为用户 {user_id} 获取推荐 (数量: {count}, 偏移: {offset})")
        
        # 从缓存获取用户数据
        user_data = None
        cache_key = f"user_data:{user_id}"
        
        if self.cache_manager:
            user_data = self.cache_manager.get(cache_key)
        
        # 如果缓存中没有，从数据库获取
        if not user_data:
            user_data = self._get_user_data_from_db(user_id)
            
            # 缓存用户数据，设置1小时过期
            if self.cache_manager and user_data:
                self.cache_manager.set(cache_key, user_data, ttl=3600)
        
        # 如果无法获取用户数据，返回兜底推荐
        if not user_data:
            self.logger.warning(f"无法获取用户 {user_id} 的数据，使用兜底推荐")
            return self.get_fallback_recommendations(count)
        
        # 添加排除项
        if excluded_items:
            if 'excluded_posts' not in user_data:
                user_data['excluded_posts'] = set()
            user_data['excluded_posts'].update(excluded_items)
        
        # 计算扩展级别，基于偏移量
        # 偏移量越大，扩展级别越高，推荐越多样化
        expansion_level = min(5, offset // (count * 2))
        
        # 根据参数调整相似度阈值和多样性权重
        if similarity_threshold is not None:
            user_data['similarity_threshold'] = similarity_threshold
        
        if diversity_weight is not None:
            user_data['diversity_weight'] = diversity_weight
        
        # 获取推荐
        if offset > count * 10:  # 如果已经翻了10页以上，使用扩展推荐
            recommendations = self.get_expanded_recommendations(
                user_data, 
                top_n=count,
                expansion_level=expansion_level
            )
        else:
            # 正常推荐
            recommendations = self.get_recommendations(user_data, top_n=count)
        
        # 如果推荐数量不足，使用兜底推荐补充
        if len(recommendations) < count:
            self.logger.info(f"推荐数量不足 ({len(recommendations)}/{count})，使用兜底推荐补充")
            fallback_count = count - len(recommendations)
            fallback_recs = self.get_fallback_recommendations(fallback_count)
            
            # 确保兜底推荐不包含已推荐的内容
            recommended_ids = {rec['post_id'] for rec in recommendations}
            unique_fallbacks = [rec for rec in fallback_recs 
                               if rec['post_id'] not in recommended_ids]
            
            recommendations.extend(unique_fallbacks)
        
        # 记录执行时间
        execution_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"用户 {user_id} 推荐生成完成，耗时: {execution_time:.4f}秒")
        
        return recommendations[:count]
    
    def _get_user_data_from_db(self, user_id):
        """从数据库获取用户数据
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户数据字典
        """
        if not self.db_pool:
            self.logger.error("数据库连接池未初始化")
            return None
        
        try:
            conn = self.db_pool.get_connection()
            user_data = {
                'user_id': user_id,
                'liked_posts': [],
                'viewed_posts': [],
                'interest_tags': [],
                'excluded_posts': set()
            }
            
            try:
                cursor = conn.cursor()
                
                # 获取用户点赞的帖子
                cursor.execute("""
                    SELECT post_id FROM user_likes 
                    WHERE user_id = %s AND created_at > DATE_SUB(NOW(), INTERVAL 30 DAY)
                    ORDER BY created_at DESC
                """, (user_id,))
                user_data['liked_posts'] = [row[0] for row in cursor.fetchall()]
                
                # 获取用户浏览的帖子
                cursor.execute("""
                    SELECT post_id FROM user_views 
                    WHERE user_id = %s AND created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
                    ORDER BY created_at DESC
                """, (user_id,))
                user_data['viewed_posts'] = [row[0] for row in cursor.fetchall()]
                user_data['excluded_posts'].update(user_data['viewed_posts'])
                
                # 获取用户兴趣标签
                cursor.execute("""
                    SELECT tag_name FROM user_interests 
                    WHERE user_id = %s
                    ORDER BY weight DESC
                """, (user_id,))
                user_data['interest_tags'] = [row[0] for row in cursor.fetchall()]
                
                cursor.close()
                return user_data
            finally:
                self.db_pool.release_connection(conn)
        except Exception as e:
            self.logger.error(f"获取用户 {user_id} 数据时出错: {e}")
            return None
            
    def save_model(self):
        """保存模型"""
        os.makedirs(self.model_dir, exist_ok=True)
        self.recommender.save_model(self.model_path)
        self.logger.info(f"模型已保存到: {self.model_path}")
        
    def load_model(self) -> bool:
        """加载模型"""
        try:
            if self.recommender.load_model(self.model_path):
                self.model_loaded = True
                self.logger.info("模型加载成功")
                return True
            else:
                self.logger.warning("模型文件不存在或损坏")
                return False
        except Exception as e:
            self.logger.error(f"加载模型时出错: {e}")
            return False 