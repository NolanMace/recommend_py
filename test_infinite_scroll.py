#!/usr/bin/env python3
"""
无限滚动推荐测试脚本

此脚本模拟用户不断向下滚动获取推荐的场景，测试推荐系统的无限滚动功能。
"""

import sys
import os
import time
import logging
import json
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommender.engine import RecommendationEngine
from cache.cache_manager import CacheManager
from utils.config_manager import ConfigManager
from exposure.exposure_pool import ExposurePoolManager
from database.db_pool import DatabasePool

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("test_infinite_scroll")

def setup_test_environment():
    """初始化测试环境"""
    logger.info("初始化测试环境...")
    
    # 加载配置
    config = ConfigManager().get_config()
    
    # 初始化组件
    db_pool = DatabasePool()
    cache_manager = CacheManager()
    recommendation_engine = RecommendationEngine(db_pool, cache_manager)
    exposure_pool_manager = ExposurePoolManager(db_pool, cache_manager)
    
    return {
        'config': config,
        'db_pool': db_pool,
        'cache_manager': cache_manager,
        'recommendation_engine': recommendation_engine,
        'exposure_pool_manager': exposure_pool_manager
    }

def simulate_infinite_scroll(env, user_id, total_pages=20, page_size=10):
    """模拟无限滚动场景
    
    Args:
        env: 测试环境
        user_id: 用户ID
        total_pages: 模拟滚动的总页数
        page_size: 每页数量
    """
    logger.info(f"开始模拟用户 {user_id} 的无限滚动，共 {total_pages} 页，每页 {page_size} 条")
    
    recommendation_engine = env['recommendation_engine']
    viewed_items = set()
    all_recommendations = []
    
    # 模拟分页请求
    for page in range(1, total_pages + 1):
        logger.info(f"获取第 {page} 页推荐...")
        
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 获取推荐
        start_time = time.time()
        recommendations = recommendation_engine.get_user_recommendations(
            user_id,
            count=page_size,
            offset=offset,
            excluded_items=viewed_items
        )
        execution_time = time.time() - start_time
        
        # 更新已浏览列表
        for item in recommendations:
            viewed_items.add(item['post_id'])
        
        # 记录结果
        logger.info(f"第 {page} 页获取了 {len(recommendations)} 条推荐，耗时: {execution_time:.4f}秒")
        
        # 分析推荐来源
        sources = {}
        for item in recommendations:
            source = item.get('source', 'default')
            if source not in sources:
                sources[source] = 0
            sources[source] += 1
        
        logger.info(f"推荐来源分布: {sources}")
        
        # 保存推荐结果
        all_recommendations.extend(recommendations)
        
        # 模拟用户浏览时间
        time.sleep(0.5)
    
    # 分析整体推荐结果
    analyze_recommendations(all_recommendations)
    
    return all_recommendations

def analyze_recommendations(recommendations):
    """分析推荐结果
    
    Args:
        recommendations: 推荐列表
    """
    if not recommendations:
        logger.warning("没有推荐结果可分析")
        return
    
    # 统计推荐来源
    sources = {}
    for item in recommendations:
        source = item.get('source', 'default')
        if source not in sources:
            sources[source] = 0
        sources[source] += 1
    
    # 统计推荐算法
    algorithms = {}
    for item in recommendations:
        algorithm = item.get('algorithm', 'unknown')
        if algorithm not in algorithms:
            algorithms[algorithm] = 0
        algorithms[algorithm] += 1
    
    # 统计推荐分数分布
    scores = [item.get('score', 0) for item in recommendations if 'score' in item]
    if scores:
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
    else:
        avg_score = max_score = min_score = 0
    
    # 输出分析结果
    logger.info(f"推荐总数: {len(recommendations)}")
    logger.info(f"推荐来源分布: {sources}")
    logger.info(f"推荐算法分布: {algorithms}")
    logger.info(f"推荐分数: 平均={avg_score:.4f}, 最高={max_score:.4f}, 最低={min_score:.4f}")
    
    # 检查是否有重复推荐
    post_ids = [item['post_id'] for item in recommendations]
    unique_ids = set(post_ids)
    if len(unique_ids) < len(post_ids):
        logger.warning(f"存在重复推荐! 总数={len(post_ids)}, 唯一数={len(unique_ids)}")
    else:
        logger.info("推荐结果中没有重复项")

def save_results(recommendations, user_id):
    """保存测试结果到文件
    
    Args:
        recommendations: 推荐列表
        user_id: 用户ID
    """
    # 创建logs目录
    os.makedirs('logs', exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"logs/infinite_scroll_test_{user_id}_{timestamp}.json"
    
    # 保存结果
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            'user_id': user_id,
            'timestamp': timestamp,
            'recommendations': recommendations
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"测试结果已保存到: {filename}")

def main():
    """主函数"""
    # 初始化测试环境
    env = setup_test_environment()
    
    # 测试用户ID
    test_user_id = 1001
    
    # 模拟无限滚动
    recommendations = simulate_infinite_scroll(
        env, 
        user_id=test_user_id,
        total_pages=15,  # 模拟15页
        page_size=10     # 每页10条
    )
    
    # 保存测试结果
    save_results(recommendations, test_user_id)
    
    logger.info("测试完成")

if __name__ == "__main__":
    main() 