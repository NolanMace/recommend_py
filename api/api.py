from flask import Flask, request, jsonify
import logging
import os
import sys
import threading
import time
from typing import Dict, List, Any, Optional, Set

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目组件
from recommender.engine import RecommendationEngine
from exposure.pool_manager import ExposurePoolManager
from hot_topics.generator import HotTopicGenerator
from cache.cache_manager import get_cache_manager
from database import db_manager

# 创建Flask应用
app = Flask(__name__)

# 创建日志记录器
logger = logging.getLogger('api_service')

# 全局组件存储
components = {}

# 用户已浏览内容缓存
user_viewed_items = {}
view_lock = threading.Lock()

def init_api_components():
    """初始化API服务所需的组件"""
    global components
    
    try:
        # 初始化缓存管理器
        cache_manager = get_cache_manager()
        components['cache_manager'] = cache_manager
        
        # 初始化推荐引擎
        recommendation_engine = RecommendationEngine(model_dir='./models')
        components['recommendation_engine'] = recommendation_engine
        
        # 初始化曝光池管理器
        exposure_manager = ExposurePoolManager(db_manager=db_manager)
        components['exposure_manager'] = exposure_manager
        
        # 初始化热点生成器
        hot_topic_generator = HotTopicGenerator(
            db_manager=db_manager,
            cache_manager=cache_manager
        )
        components['hot_topic_generator'] = hot_topic_generator
        
        logger.info("API组件初始化完成")
        return True
    except Exception as e:
        logger.error(f"API组件初始化失败: {e}")
        return False

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """获取用户推荐内容
    
    支持无限滚动和分页加载
    
    查询参数:
        user_id: 用户ID
        page: 页码(从1开始)
        page_size: 每页条数(默认20)
        viewed_items: 已浏览内容IDs(可选,逗号分隔)
    """
    try:
        # 获取请求参数
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'error': '缺少user_id参数'}), 400
            
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 20))
        
        # 限制page_size大小，防止请求过大数据量
        if page_size > 50:
            page_size = 50
        
        # 获取用户已浏览内容
        viewed_items_param = request.args.get('viewed_items', '')
        if viewed_items_param:
            # 如果请求中包含已浏览内容，先更新记录
            new_viewed_items = set(int(item) for item in viewed_items_param.split(',') if item.isdigit())
            
            with view_lock:
                if user_id not in user_viewed_items:
                    user_viewed_items[user_id] = set()
                user_viewed_items[user_id].update(new_viewed_items)
        
        # 从缓存中获取已浏览内容
        viewed_set = set()
        with view_lock:
            if user_id in user_viewed_items:
                viewed_set = user_viewed_items[user_id].copy()
        
        # 从缓存系统获取更多已浏览内容
        cache_manager = components['cache_manager']
        viewed_key = f"user_viewed:{user_id}"
        cached_viewed = cache_manager.get(viewed_key)
        if cached_viewed:
            viewed_set.update(cached_viewed)
        
        # 获取推荐引擎
        recommendation_engine = components['recommendation_engine']
        
        # 根据页码计算偏移量
        offset = (page - 1) * page_size
        
        # 根据浏览深度调整推荐策略
        recommendations = []
        
        # 获取用户数据
        user_data = db_manager.get_user_data(user_id)
        
        # 添加浏览历史到用户数据
        user_data['viewed_posts'] = list(viewed_set)
        
        # 分层次获取推荐 - 根据浏览深度调整策略
        if page <= 3:  
            # 前3页使用严格个性化推荐
            recommendations = get_personalized_recommendations(
                recommendation_engine, 
                user_data, 
                page_size, 
                viewed_set
            )
        elif page <= 10:  
            # 4-10页使用混合推荐
            recommendations = get_mixed_recommendations(
                recommendation_engine, 
                user_data, 
                page_size, 
                viewed_set, 
                personalized_weight=0.7, 
                trending_weight=0.3
            )
        else:  
            # 10页以后使用扩展推荐
            recommendations = get_extended_recommendations(
                recommendation_engine, 
                user_data, 
                page_size, 
                viewed_set,
                expansion_level=(page-10)//5 + 1  # 随页数增加扩展级别
            )
        
        # 如果推荐结果数量仍不足，补充兜底推荐
        if len(recommendations) < page_size:
            fallback_recs = get_fallback_recommendations(
                user_data, 
                page_size - len(recommendations), 
                viewed_set
            )
            recommendations.extend(fallback_recs)
        
        # 记录新浏览的内容
        new_viewed = [item['post_id'] for item in recommendations]
        viewed_set.update(new_viewed)
        
        # 更新内存中的浏览历史
        with view_lock:
            if user_id not in user_viewed_items:
                user_viewed_items[user_id] = set()
            user_viewed_items[user_id].update(new_viewed)
        
        # 更新缓存中的浏览历史，过期时间7天
        cache_manager.set(viewed_key, viewed_set, ttl=86400*7)
        
        # 构建响应
        response = {
            'page': page,
            'page_size': page_size,
            'total_viewed': len(viewed_set),
            'has_more': len(recommendations) == page_size,  # 判断是否还有更多
            'recommendations': recommendations
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"获取推荐时出错: {e}")
        return jsonify({'error': f'获取推荐失败: {str(e)}'}), 500

def get_personalized_recommendations(engine, user_data, count, exclude_ids=None):
    """获取个性化推荐结果"""
    try:
        # 复制用户数据以避免修改原始数据
        data = user_data.copy()
        
        # 添加排除项
        if exclude_ids:
            if 'viewed_posts' not in data:
                data['viewed_posts'] = list(exclude_ids)
            else:
                # 确保不重复添加
                existing = set(data['viewed_posts'])
                for item_id in exclude_ids:
                    if item_id not in existing:
                        data['viewed_posts'].append(item_id)
        
        # 获取推荐
        recommendations = engine.get_recommendations(data, top_n=count)
        return recommendations
    except Exception as e:
        logger.error(f"获取个性化推荐失败: {e}")
        return []

def get_mixed_recommendations(engine, user_data, count, exclude_ids=None, 
                             personalized_weight=0.6, trending_weight=0.4):
    """获取混合推荐结果(个性化+热门混合)"""
    try:
        # 计算各部分的数量
        personalized_count = int(count * personalized_weight)
        trending_count = count - personalized_count
        
        # 先获取个性化推荐
        personalized = get_personalized_recommendations(
            engine, user_data, personalized_count, exclude_ids
        )
        
        # 添加到排除列表
        if exclude_ids is None:
            exclude_ids = set()
        else:
            exclude_ids = set(exclude_ids)
            
        for item in personalized:
            exclude_ids.add(item['post_id'])
        
        # 获取热门推荐来补充
        hot_posts = db_manager.get_hot_posts(
            limit=trending_count, 
            exclude_ids=list(exclude_ids)
        )
        
        # 将热门转换为推荐格式
        trending = []
        for post in hot_posts:
            trending.append({
                'post_id': post['post_id'],
                'title': post['title'],
                'score': post.get('heat_score', 0.5),
                'algorithm': 'trending'
            })
        
        # 合并结果
        recommendations = personalized + trending
        return recommendations
    except Exception as e:
        logger.error(f"获取混合推荐失败: {e}")
        return []

def get_extended_recommendations(engine, user_data, count, exclude_ids=None, expansion_level=1):
    """获取扩展推荐结果(扩大推荐范围)"""
    try:
        # 复制用户数据
        data = user_data.copy()
        
        # 根据扩展级别降低相似度阈值(模拟)
        # 实际实现中,推荐引擎需要支持相似度阈值参数
        # 这里我们通过增加推荐数量再截取的方式模拟
        expanded_count = count * (1 + expansion_level * 0.5)
        expanded_count = int(min(expanded_count, count * 3))  # 最多扩大3倍
        
        # 获取推荐
        recommendations = get_personalized_recommendations(
            engine, data, expanded_count, exclude_ids
        )
        
        # 如果结果仍然不足，加入分类扩展
        if len(recommendations) < count and 'interest_tags' in data:
            # 获取用户可能感兴趣的相关标签(分类扩展)
            related_tags = db_manager.get_related_tags(data['interest_tags'])
            
            # 临时添加这些标签到用户兴趣中
            original_tags = data.get('interest_tags', [])
            data['interest_tags'] = original_tags + related_tags
            
            # 再次获取推荐
            more_recommendations = get_personalized_recommendations(
                engine, data, count - len(recommendations), exclude_ids
            )
            
            # 标记这些推荐为扩展推荐
            for rec in more_recommendations:
                rec['algorithm'] = f"{rec.get('algorithm', 'unknown')}_expanded"
                
            # 合并结果
            recommendations.extend(more_recommendations)
        
        # 截取需要的数量
        return recommendations[:count]
    except Exception as e:
        logger.error(f"获取扩展推荐失败: {e}")
        return []

def get_fallback_recommendations(user_data, count, exclude_ids=None):
    """获取兜底推荐(当其他策略都不足时)"""
    try:
        # 使用多种兜底策略
        if exclude_ids is None:
            exclude_ids = set()
        else:
            exclude_ids = set(exclude_ids)
            
        # 策略1: 最新内容(时间最新的内容)
        new_posts = db_manager.get_newest_posts(
            limit=count, 
            exclude_ids=list(exclude_ids)
        )
        
        # 转换为推荐格式
        recommendations = []
        for post in new_posts:
            exclude_ids.add(post['post_id'])  # 防止后续策略重复
            recommendations.append({
                'post_id': post['post_id'],
                'title': post['title'],
                'score': 0.3,  # 兜底推荐得分较低
                'algorithm': 'newest'
            })
            
        # 如果还不够，使用随机高质量内容补充
        if len(recommendations) < count:
            quality_posts = db_manager.get_quality_posts(
                limit=count - len(recommendations),
                exclude_ids=list(exclude_ids)
            )
            
            for post in quality_posts:
                recommendations.append({
                    'post_id': post['post_id'],
                    'title': post['title'],
                    'score': 0.2,
                    'algorithm': 'quality'
                })
                
        return recommendations
    except Exception as e:
        logger.error(f"获取兜底推荐失败: {e}")
        return []

@app.route('/api/hot_topics', methods=['GET'])
def get_hot_topics():
    """获取热门话题"""
    try:
        count = int(request.args.get('count', 10))
        if count > 50:
            count = 50  # 限制最大数量
            
        hot_topic_generator = components['hot_topic_generator']
        hot_topics = hot_topic_generator.get_hot_topics(limit=count)
        
        return jsonify({
            'count': len(hot_topics),
            'hot_topics': hot_topics
        })
    except Exception as e:
        logger.error(f"获取热门话题时出错: {e}")
        return jsonify({'error': f'获取热门话题失败: {str(e)}'}), 500

@app.route('/api/mark_viewed', methods=['POST'])
def mark_items_as_viewed():
    """标记内容为已浏览"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '无效的请求数据'}), 400
            
        user_id = data.get('user_id')
        item_ids = data.get('item_ids', [])
        
        if not user_id or not item_ids:
            return jsonify({'error': '缺少必要参数'}), 400
            
        # 更新内存中的浏览历史
        with view_lock:
            if user_id not in user_viewed_items:
                user_viewed_items[user_id] = set()
            user_viewed_items[user_id].update(item_ids)
        
        # 更新缓存中的浏览历史
        cache_manager = components['cache_manager']
        viewed_key = f"user_viewed:{user_id}"
        current_viewed = cache_manager.get(viewed_key) or set()
        current_viewed.update(item_ids)
        cache_manager.set(viewed_key, current_viewed, ttl=86400*7)  # 7天过期
        
        # 记录浏览历史到数据库
        db_manager.record_user_views(user_id, item_ids)
        
        return jsonify({
            'success': True,
            'message': f'已更新{len(item_ids)}条浏览历史',
            'total_viewed': len(current_viewed)
        })
    except Exception as e:
        logger.error(f"标记已浏览内容时出错: {e}")
        return jsonify({'error': f'标记已浏览失败: {str(e)}'}), 500

@app.route('/api/cleanup_cache', methods=['POST'])
def cleanup_cache():
    """清理缓存"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'error': '缺少user_id参数'}), 400
            
        # 清理内存缓存
        with view_lock:
            if user_id in user_viewed_items:
                del user_viewed_items[user_id]
                
        # 清理Redis缓存
        cache_manager = components['cache_manager']
        viewed_key = f"user_viewed:{user_id}"
        cache_manager.delete(viewed_key)
        
        return jsonify({
            'success': True,
            'message': f'已清理用户{user_id}的浏览缓存'
        })
    except Exception as e:
        logger.error(f"清理缓存时出错: {e}")
        return jsonify({'error': f'清理缓存失败: {str(e)}'}), 500

# 启动API服务
def start_api_server(host='0.0.0.0', port=5000, debug=False):
    """启动API服务器"""
    # 初始化组件
    if not init_api_components():
        logger.error("API组件初始化失败，无法启动服务")
        return
        
    logger.info(f"API服务器启动中，监听: {host}:{port}")
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='推荐系统API服务')
    parser.add_argument('--host', default='0.0.0.0', help='API服务器监听主机')
    parser.add_argument('--port', type=int, default=5000, help='API服务器监听端口')
    parser.add_argument('--debug', action='store_true', help='是否启用调试模式')
    
    args = parser.parse_args()
    
    # 启动API服务器
    start_api_server(host=args.host, port=args.port, debug=args.debug) 