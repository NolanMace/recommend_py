from flask import Blueprint, request, jsonify
import logging
import time
import traceback
import sys
import os

# 添加父目录到系统路径，以便导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.engine import RecommendationEngine
from cache.cache_manager import CacheManager
from utils.config_manager import ConfigManager
from exposure.exposure_pool import ExposurePoolManager
from database.db_pool import DatabasePool

# 初始化日志
logger = logging.getLogger(__name__)

# 创建蓝图
recommendation_bp = Blueprint('recommendations', __name__)

# 加载配置
config = ConfigManager().get_config()

# 初始化组件
cache_manager = CacheManager()
db_pool = DatabasePool()
recommendation_engine = RecommendationEngine(db_pool, cache_manager)
exposure_pool_manager = ExposurePoolManager(db_pool, cache_manager)

# 用户浏览历史缓存键格式
USER_VIEWED_CACHE_KEY = "user:{0}:viewed_items"

@recommendation_bp.route('/recommendations', methods=['GET'])
def get_recommendations():
    """获取用户推荐
    
    支持分页和无限滚动，根据页码动态调整推荐策略
    
    参数:
        user_id (str): 用户ID
        page (int, optional): 页码，默认为1
        page_size (int, optional): 每页数量，默认为20
    
    返回:
        JSON: 包含推荐列表和是否有更多内容的标志
    """
    start_time = time.time()
    
    try:
        # 获取请求参数
        user_id = request.args.get('user_id')
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 20))
        
        if not user_id:
            return jsonify({"error": "缺少必要参数: user_id"}), 400
        
        # 获取用户已浏览内容列表
        viewed_cache_key = USER_VIEWED_CACHE_KEY.format(user_id)
        viewed_items = cache_manager.get(viewed_cache_key) or set()
        
        # 根据页码确定推荐策略
        if page <= 3:
            # 前3页: 严格个性化推荐
            similarity_threshold = config.get('recommender.similarity_threshold', 0.7)
            diversity_weight = config.get('recommender.diversity_weight', 0.2)
        elif page <= 10:
            # 4-10页: 混合推荐(个性化+热门)
            similarity_threshold = config.get('recommender.similarity_threshold', 0.7) * 0.8
            diversity_weight = config.get('recommender.diversity_weight', 0.2) * 1.5
        else:
            # 10页后: 扩展推荐(降低相似度阈值)
            similarity_threshold = config.get('recommender.similarity_threshold', 0.7) * 0.6
            diversity_weight = config.get('recommender.diversity_weight', 0.2) * 2.0
        
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 从推荐引擎获取推荐
        recommendations = recommendation_engine.get_user_recommendations(
            user_id, 
            count=page_size, 
            offset=offset,
            similarity_threshold=similarity_threshold,
            diversity_weight=diversity_weight,
            excluded_items=viewed_items
        )
        
        # 检查是否获取足够的推荐
        if len(recommendations) < page_size:
            # 推荐数量不足，使用曝光池补充
            logger.info(f"推荐数量不足 ({len(recommendations)}/{page_size})，使用曝光池补充")
            
            # 计算需要补充的数量
            needed_count = page_size - len(recommendations)
            
            # 已推荐的内容ID列表
            recommended_ids = [item['post_id'] for item in recommendations]
            
            # 需要排除的内容: 已推荐的 + 已浏览的
            excluded_ids = set(recommended_ids) | viewed_items
            
            # 根据页码调整曝光池策略
            if page <= 5:
                # 前5页主要补充热门内容
                pool_weights = {
                    'hot': 0.6, 
                    'new': 0.3, 
                    'quality': 0.1
                }
            elif page <= 10:
                # 5-10页平衡各类内容
                pool_weights = {
                    'hot': 0.4,
                    'new': 0.4,
                    'quality': 0.2
                }
            else:
                # 10页后增加新内容比例
                pool_weights = {
                    'hot': 0.3,
                    'new': 0.5,
                    'quality': 0.2
                }
            
            # 从曝光池获取补充内容
            fallback_items = exposure_pool_manager.get_mixed_items(
                needed_count,
                excluded_ids=excluded_ids,
                pool_weights=pool_weights
            )
            
            # 将补充内容添加到推荐结果中
            recommendations.extend(fallback_items)
        
        # 更新缓存中的已浏览列表
        for item in recommendations:
            viewed_items.add(item['post_id'])
        
        # 存储已浏览列表，设置7天过期时间
        cache_manager.set(viewed_cache_key, viewed_items, ttl=7*24*60*60)
        
        # 准备响应
        response = {
            "recommendations": recommendations,
            "has_more": True,  # 总是返回True，支持无限滚动
            "page": page,
            "page_size": page_size,
            "count": len(recommendations)
        }
        
        # 记录执行时间
        execution_time = time.time() - start_time
        logger.info(f"用户 {user_id} 获取推荐 (页码: {page}, 数量: {len(recommendations)}) 耗时: {execution_time:.4f}秒")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"获取推荐失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "获取推荐时发生错误", "details": str(e)}), 500


@recommendation_bp.route('/mark_viewed', methods=['POST'])
def mark_viewed():
    """标记内容为已浏览
    
    将内容添加到用户的已浏览列表
    
    参数:
        user_id (str): 用户ID
        item_ids (list): 内容ID列表
    
    返回:
        JSON: 操作结果
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "无效的请求数据"}), 400
        
        user_id = data.get('user_id')
        item_ids = data.get('item_ids', [])
        
        if not user_id:
            return jsonify({"error": "缺少必要参数: user_id"}), 400
        
        if not item_ids:
            return jsonify({"error": "缺少必要参数: item_ids"}), 400
        
        # 获取用户已浏览内容列表
        viewed_cache_key = USER_VIEWED_CACHE_KEY.format(user_id)
        viewed_items = cache_manager.get(viewed_cache_key) or set()
        
        # 添加新的已浏览内容
        viewed_items.update(item_ids)
        
        # 更新缓存，设置7天过期时间
        cache_manager.set(viewed_cache_key, viewed_items, ttl=7*24*60*60)
        
        return jsonify({
            "success": True,
            "viewed_count": len(viewed_items)
        })
    
    except Exception as e:
        logger.error(f"标记已浏览内容失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "标记已浏览内容时发生错误", "details": str(e)}), 500


@recommendation_bp.route('/hot_topics', methods=['GET'])
def get_hot_topics():
    """获取热门话题
    
    返回系统当前的热门话题列表
    
    参数:
        count (int, optional): 返回的热门话题数量，默认为10
    
    返回:
        JSON: 热门话题列表
    """
    try:
        # 获取请求参数
        count = int(request.args.get('count', 10))
        
        # 从缓存中获取热门话题
        hot_topics = cache_manager.get("hot_topics")
        
        if not hot_topics:
            # 如果缓存中没有，从数据库获取
            conn = db_pool.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT topic_id, topic_name, weight, update_time 
                    FROM hot_topics 
                    ORDER BY weight DESC, update_time DESC 
                    LIMIT %s
                """, (count,))
                hot_topics = [
                    {
                        "topic_id": row[0],
                        "topic_name": row[1],
                        "weight": float(row[2]),
                        "update_time": row[3].isoformat() if hasattr(row[3], 'isoformat') else row[3]
                    }
                    for row in cursor.fetchall()
                ]
                cursor.close()
                
                # 存入缓存，设置5分钟过期
                if hot_topics:
                    cache_manager.set("hot_topics", hot_topics, ttl=5*60)
            finally:
                db_pool.release_connection(conn)
        
        return jsonify({
            "hot_topics": hot_topics or [],
            "count": len(hot_topics) if hot_topics else 0
        })
    
    except Exception as e:
        logger.error(f"获取热门话题失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "获取热门话题时发生错误", "details": str(e)}), 500 