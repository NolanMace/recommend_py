#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import json
import time
from typing import List, Dict, Any
import argparse

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_recommender")

# 导入推荐系统组件
from utils.config_manager import ConfigManager
from database.db_manager import get_db_connection, get_user_data, get_active_users
from recommender.engine import RecommendationEngine
from exposure.pool_manager import ExposurePoolManager
from cache.cache_manager import CacheManager

def init_components():
    """初始化组件"""
    logger.info("开始初始化组件...")
    
    # 初始化配置管理器并加载配置
    config = ConfigManager().get_config()
    logger.info("配置加载成功")
    
    # 测试数据库连接
    try:
        conn = get_db_connection()
        logger.info(f"数据库连接成功: {conn.host}")
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        sys.exit(1)
    
    # 初始化缓存管理器
    cache_mgr = CacheManager()
    logger.info(f"缓存管理器初始化成功，当前缓存项数: {len(cache_mgr.cache)}")
    
    # 初始化推荐引擎
    engine = RecommendationEngine()
    logger.info("推荐引擎初始化成功")
    
    # 初始化曝光池管理器
    pool_mgr = ExposurePoolManager()
    logger.info("曝光池管理器初始化成功")
    
    logger.info("所有组件初始化完成")
    return config, cache_mgr, engine, pool_mgr

def test_user_recommendation(user_id: str, engine: RecommendationEngine, count: int = 10):
    """测试为特定用户生成推荐"""
    logger.info(f"正在为用户 {user_id} 生成 {count} 条推荐...")
    
    try:
        # 获取用户数据
        user_data = get_user_data(user_id)
        if not user_data:
            logger.warning(f"未找到用户 {user_id} 的数据，将使用默认推荐")
        
        # 生成推荐
        start_time = time.time()
        recommendations = engine.generate_recommendations(user_id, count=count)
        end_time = time.time()
        
        # 打印结果
        logger.info(f"成功为用户 {user_id} 生成 {len(recommendations)} 条推荐，耗时: {end_time - start_time:.4f}秒")
        for i, rec in enumerate(recommendations[:count], 1):
            logger.info(f"{i}. 帖子ID: {rec['post_id']}, 标题: {rec.get('title', '无标题')[:30]}, 得分: {rec.get('score', 0):.4f}")
        
        return recommendations
    except Exception as e:
        logger.error(f"生成推荐时出错: {e}")
        return []

def test_batch_recommendations(engine: RecommendationEngine, user_count: int = 5, rec_count: int = 10):
    """测试批量生成推荐"""
    logger.info(f"正在为 {user_count} 位活跃用户生成批量推荐...")
    
    try:
        # 获取活跃用户
        active_users = get_active_users(days=7, limit=user_count)
        if not active_users:
            logger.warning("未找到活跃用户，无法测试批量推荐")
            return
        
        user_ids = [user['user_id'] for user in active_users]
        logger.info(f"找到 {len(user_ids)} 位活跃用户")
        
        # 批量生成推荐
        start_time = time.time()
        results = {}
        
        for user_id in user_ids:
            recommendations = engine.generate_recommendations(user_id, count=rec_count)
            results[user_id] = recommendations
            logger.info(f"用户 {user_id}: 生成 {len(recommendations)} 条推荐")
        
        end_time = time.time()
        avg_time = (end_time - start_time) / len(user_ids) if user_ids else 0
        
        logger.info(f"批量推荐完成，平均每位用户耗时: {avg_time:.4f}秒")
        return results
    except Exception as e:
        logger.error(f"批量生成推荐时出错: {e}")
        return {}

def test_pool_refresh(pool_mgr: ExposurePoolManager):
    """测试刷新曝光池"""
    logger.info("开始测试曝光池刷新...")
    
    try:
        start_time = time.time()
        refresh_result = pool_mgr.refresh_all_pools()
        end_time = time.time()
        
        if refresh_result:
            logger.info(f"曝光池刷新成功，耗时: {end_time - start_time:.4f}秒")
            for pool_name, count in refresh_result.items():
                logger.info(f"曝光池 '{pool_name}': {count} 条内容")
        else:
            logger.warning("曝光池刷新返回空结果")
        
        return refresh_result
    except Exception as e:
        logger.error(f"刷新曝光池时出错: {e}")
        return {}

def save_results_to_file(data: Any, filename: str):
    """将结果保存到文件"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到文件: {filename}")
    except Exception as e:
        logger.error(f"保存结果到文件时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='推荐系统测试工具')
    parser.add_argument('--user', type=str, help='指定用户ID进行测试')
    parser.add_argument('--count', type=int, default=10, help='推荐数量')
    parser.add_argument('--batch', type=int, help='批量测试的用户数量')
    parser.add_argument('--pool', action='store_true', help='测试曝光池刷新')
    parser.add_argument('--save', action='store_true', help='保存结果到文件')
    
    args = parser.parse_args()
    
    # 初始化组件
    config, cache_mgr, engine, pool_mgr = init_components()
    
    results = {}
    
    # 测试单用户推荐
    if args.user:
        results['user_recommendations'] = test_user_recommendation(args.user, engine, args.count)
    
    # 测试批量推荐
    if args.batch:
        results['batch_recommendations'] = test_batch_recommendations(engine, args.batch, args.count)
    
    # 测试曝光池刷新
    if args.pool:
        results['pool_refresh'] = test_pool_refresh(pool_mgr)
    
    # 如果没有指定测试类型，运行所有测试
    if not (args.user or args.batch or args.pool):
        logger.info("未指定测试类型，将运行默认测试...")
        # 默认测试一个随机用户
        active_users = get_active_users(days=7, limit=1)
        if active_users:
            test_user = active_users[0]['user_id']
            results['user_recommendations'] = test_user_recommendation(test_user, engine, args.count)
        else:
            logger.warning("未找到活跃用户，将使用默认用户ID")
            results['user_recommendations'] = test_user_recommendation("default_user", engine, args.count)
        
        # 默认测试曝光池
        results['pool_refresh'] = test_pool_refresh(pool_mgr)
    
    # 保存结果到文件
    if args.save and results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_results_to_file(results, f"recommend_test_{timestamp}.json")

if __name__ == '__main__':
    main() 