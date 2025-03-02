#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
迁移脚本：将推荐系统从API服务迁移到数据库存储模式
"""
import os
import sys
import argparse
import json
from datetime import datetime
import time
import threading

from database import get_db_pool
from recommender import Recommender, FeatureProcessor
from config import DATABASE_CONFIG, HOT_TOPICS_CONFIG

# 数据库连接池
db_pool = get_db_pool()

# 全局推荐器实例
recommender = None
recommender_lock = threading.Lock()

def init_recommend_logs_table():
    """初始化推荐日志表"""
    sql = """
    CREATE TABLE IF NOT EXISTS recommend_logs (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        user_id BIGINT NOT NULL COMMENT '用户ID',
        recommend_time DATETIME NOT NULL COMMENT '推荐时间',
        post_count INT NOT NULL DEFAULT 0 COMMENT '推荐帖子数量',
        process_time INT NOT NULL DEFAULT 0 COMMENT '处理时间(毫秒)',
        INDEX idx_user_time (user_id, recommend_time)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='推荐日志表'
    """
    db_pool.execute(sql)
    print(f"[{datetime.now()}] 推荐日志表初始化成功")

def get_recommender():
    """获取推荐器实例（懒加载）"""
    global recommender
    if recommender is None:
        with recommender_lock:
            if recommender is None:
                print(f"[{datetime.now()}] 初始化推荐器...")
                fp = FeatureProcessor()
                fp.load_data()
                recommender = Recommender(fp)
    return recommender

def generate_recommendations_for_all_active_users():
    """为所有活跃用户生成推荐结果"""
    # 查询最近活跃的用户
    active_days = DATABASE_CONFIG['active_user_days']
    sql = """
    SELECT DISTINCT b.user_id 
    FROM user_behavior b
    WHERE b.timestamp > DATE_SUB(NOW(), INTERVAL %s DAY)
    """
    active_users = db_pool.query(sql, (active_days,))
    
    total_users = len(active_users)
    print(f"[{datetime.now()}] 找到{total_users}个活跃用户")
    
    # 生成推荐结果
    rec = get_recommender()
    success_count = 0
    error_count = 0
    
    for i, row in enumerate(active_users):
        try:
            user_id = row[0] if isinstance(row, tuple) else row['user_id']
            
            # 为用户生成推荐
            start_time = time.time()
            posts = rec.hybrid_recommend(user_id, save_to_db=True)
            process_time = round((time.time() - start_time) * 1000)  # 毫秒
            
            # 显示进度
            success_count += 1
            if success_count % 10 == 0 or success_count == total_users:
                print(f"[{datetime.now()}] 进度: {success_count}/{total_users} - "
                      f"用户{user_id}生成推荐{len(posts)}条，耗时{process_time}ms")
        
        except Exception as e:
            error_count += 1
            print(f"[{datetime.now()}] 为用户{user_id}生成推荐失败: {str(e)}")
    
    print(f"[{datetime.now()}] 推荐生成完成: 成功{success_count}，失败{error_count}")
    return success_count

def generate_hot_topics():
    """生成热点话题"""
    rec = get_recommender()
    
    print(f"[{datetime.now()}] 开始生成热点话题...")
    start_time = time.time()
    hot_topics = rec.generate_hot_topics(HOT_TOPICS_CONFIG['count'])
    process_time = round((time.time() - start_time) * 1000)  # 毫秒
    
    print(f"[{datetime.now()}] 热点生成完成: {len(hot_topics)}个热点话题, 耗时{process_time}ms")
    return len(hot_topics)

def execute_many(self, sql, args_list):
    # 分批执行以提高性能
    batch_size = 1000

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='推荐系统迁移工具')
    parser.add_argument('--recommendations', action='store_true', 
                      help='为活跃用户生成推荐结果')
    parser.add_argument('--hot-topics', action='store_true', 
                      help='生成热点话题')
    parser.add_argument('--all', action='store_true', 
                      help='执行所有数据迁移任务')
    args = parser.parse_args()
    
    # 如果没有指定任何参数，显示帮助信息
    if not (args.recommendations or args.hot_topics or args.all):
        parser.print_help()
        return
    
    try:
        print(f"[{datetime.now()}] ===== 推荐系统迁移工具 =====")
        
        # 确保日志表存在
        init_recommend_logs_table()
        
        # 执行各项任务
        if args.recommendations or args.all:
            print(f"[{datetime.now()}] 开始为活跃用户生成推荐...")
            count = generate_recommendations_for_all_active_users()
            print(f"[{datetime.now()}] 推荐生成完成，共{count}个用户")
        
        if args.hot_topics or args.all:
            print(f"[{datetime.now()}] 开始生成热点话题...")
            count = generate_hot_topics()
            print(f"[{datetime.now()}] 热点生成完成，共{count}个热点")
            
        print(f"[{datetime.now()}] ===== 迁移工具执行完成 =====")
        
    except Exception as e:
        print(f"[{datetime.now()}] 执行过程中出现错误: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 