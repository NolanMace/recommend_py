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
    """混合推荐引擎，结合多种推荐算法"""
    def __init__(self):
        super().__init__("hybrid")
        self.recommenders = {
            'tfidf': TFIDFRecommender(),
            'collaborative': CollaborativeFilteringRecommender()
        }
        
        # 各算法权重
        self.weights = {
            'tfidf': 0.6,
            'collaborative': 0.4
        }
        
    def fit(self, data: Dict[str, Any]):
        """训练所有子推荐器"""
        for name, recommender in self.recommenders.items():
            self.logger.info(f"正在训练 {name} 推荐器...")
            recommender.fit(data)
            
        return self
        
    def recommend(self, user_data: Dict, top_n: int = 10) -> List[Dict]:
        """为用户生成混合推荐"""
        # 存储每个算法的推荐结果
        all_recommendations = {}
        
        # 获取每个推荐器的结果
        for name, recommender in self.recommenders.items():
            recs = recommender.recommend(user_data, top_n=top_n*2)  # 获取2倍候选集
            all_recommendations[name] = recs
        
        # 合并结果并按评分排序
        merged_recs = []
        post_added = set()  # 跟踪已添加的帖子ID，避免重复
        
        # 首先按权重合并结果
        for name, recs in all_recommendations.items():
            weight = self.weights[name]
            for rec in recs:
                post_id = rec['post_id']
                if post_id not in post_added:
                    # 调整评分乘以权重
                    rec['score'] *= weight
                    merged_recs.append(rec)
                    post_added.add(post_id)
        
        # 按评分排序并截取top_n
        merged_recs.sort(key=lambda x: x['score'], reverse=True)
        return merged_recs[:top_n]
    
    def save_model(self, path: str):
        """保存所有子推荐器模型"""
        base_dir = os.path.dirname(path)
        os.makedirs(base_dir, exist_ok=True)
        
        # 保存模型元数据
        metadata = {
            'weights': self.weights,
            'timestamp': datetime.now()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(metadata, f)
            
        # 保存各子推荐器
        for name, recommender in self.recommenders.items():
            sub_path = os.path.join(base_dir, f"{name}_model.pkl")
            recommender.save_model(sub_path)
            
        self.logger.info(f"混合推荐器模型已保存到: {base_dir}")
        
    def load_model(self, path: str):
        """加载所有子推荐器模型"""
        if not os.path.exists(path):
            self.logger.warning(f"模型文件不存在: {path}")
            return False
            
        with open(path, 'rb') as f:
            metadata = pickle.load(f)
            
        self.weights = metadata['weights']
        
        # 加载各子推荐器
        base_dir = os.path.dirname(path)
        for name, recommender in self.recommenders.items():
            sub_path = os.path.join(base_dir, f"{name}_model.pkl")
            recommender.load_model(sub_path)
            
        self.logger.info(f"混合推荐器模型已加载，时间戳: {metadata['timestamp']}")
        return True

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
        
    def get_recommendations(self, user_data: Dict, top_n: int = 10) -> List[Dict]:
        """获取用户推荐"""
        if not self.model_loaded:
            self.logger.warning("模型未加载，尝试加载模型...")
            if not self.load_model():
                self.logger.error("模型加载失败，无法提供推荐")
                return []
        
        try:
            recommendations = self.recommender.recommend(user_data, top_n)
            return recommendations
        except Exception as e:
            self.logger.error(f"生成推荐时出错: {e}")
            return []
            
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