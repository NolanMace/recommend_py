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
import logging
from typing import List, Dict, Any

from database import get_db_pool
from recommender import Recommender, FeatureProcessor
from config import DATABASE_CONFIG, HOT_TOPICS_CONFIG
from config.config_manager import get_config_manager
from database.database import get_db_manager

# 数据库连接池
db_pool = get_db_pool()

# 全局推荐器实例
recommender = None
recommender_lock = threading.Lock()

class DatabaseMigrator:
    """数据库迁移管理器
    
    负责管理数据库结构的迁移和更新
    """
    
    def __init__(self):
        self.logger = logging.getLogger("database_migrator")
        self.config_manager = get_config_manager()
        self.db_manager = get_db_manager()
        
        # 获取配置
        self.config = self.config_manager.get('database')
        
        # 迁移文件目录
        self.migrations_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'migrations'
        )
        
        # 确保迁移目录存在
        os.makedirs(self.migrations_dir, exist_ok=True)
    
    def init_migration_table(self):
        """初始化迁移记录表"""
        sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version VARCHAR(14) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            applied_at DATETIME NOT NULL,
            success BOOLEAN NOT NULL DEFAULT TRUE,
            error_message TEXT
        )
        """
        self.db_manager.execute_update(sql)
        self.logger.info("迁移记录表初始化完成")
    
    def get_applied_migrations(self) -> List[str]:
        """获取已应用的迁移版本列表"""
        sql = """
        SELECT version FROM schema_migrations 
        WHERE success = TRUE 
        ORDER BY version
        """
        results = self.db_manager.execute_query(sql)
        return [row['version'] for row in results]
    
    def get_pending_migrations(self) -> List[str]:
        """获取待执行的迁移文件列表"""
        # 获取所有迁移文件
        migration_files = []
        for file in os.listdir(self.migrations_dir):
            if file.endswith('.sql'):
                version = file.split('_')[0]
                migration_files.append((version, file))
        
        # 按版本号排序
        migration_files.sort()
        
        # 获取已应用的迁移
        applied = set(self.get_applied_migrations())
        
        # 返回未应用的迁移
        return [f for v, f in migration_files if v not in applied]
    
    def apply_migration(self, migration_file: str) -> bool:
        """应用单个迁移
        
        Args:
            migration_file: 迁移文件名
            
        Returns:
            bool: 是否成功
        """
        version = migration_file.split('_')[0]
        name = migration_file[15:-4]  # 去掉版本号和扩展名
        
        # 读取迁移文件
        file_path = os.path.join(self.migrations_dir, migration_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sql = f.read()
        except Exception as e:
            self.logger.error(f"读取迁移文件失败: {e}")
            return False
        
        # 开始执行迁移
        try:
            # 执行迁移SQL
            self.db_manager.execute_update(sql)
            
            # 记录迁移结果
            record_sql = """
            INSERT INTO schema_migrations 
            (version, name, applied_at, success)
            VALUES (%s, %s, %s, TRUE)
            """
            self.db_manager.execute_update(
                record_sql,
                (version, name, datetime.now())
            )
            
            self.logger.info(f"迁移 {migration_file} 执行成功")
            return True
            
        except Exception as e:
            # 记录失败结果
            record_sql = """
            INSERT INTO schema_migrations 
            (version, name, applied_at, success, error_message)
            VALUES (%s, %s, %s, FALSE, %s)
            """
            self.db_manager.execute_update(
                record_sql,
                (version, name, datetime.now(), str(e))
            )
            
            self.logger.error(f"迁移 {migration_file} 执行失败: {e}")
            return False
    
    def migrate(self) -> bool:
        """执行所有待执行的迁移
        
        Returns:
            bool: 是否全部成功
        """
        # 初始化迁移表
        self.init_migration_table()
        
        # 获取待执行的迁移
        pending = self.get_pending_migrations()
        if not pending:
            self.logger.info("没有待执行的迁移")
            return True
        
        # 执行迁移
        success = True
        for migration_file in pending:
            if not self.apply_migration(migration_file):
                success = False
                break
        
        return success
    
    def rollback(self, steps: int = 1) -> bool:
        """回滚指定数量的迁移
        
        Args:
            steps: 回滚步数
            
        Returns:
            bool: 是否成功
        """
        # 获取最近的迁移记录
        sql = """
        SELECT version, name 
        FROM schema_migrations 
        WHERE success = TRUE 
        ORDER BY version DESC 
        LIMIT %s
        """
        migrations = self.db_manager.execute_query(sql, (steps,))
        
        if not migrations:
            self.logger.info("没有可回滚的迁移")
            return True
        
        # 执行回滚
        success = True
        for migration in reversed(migrations):
            version = migration['version']
            name = migration['name']
            
            # 查找回滚文件
            rollback_file = f"{version}_rollback_{name}.sql"
            file_path = os.path.join(self.migrations_dir, rollback_file)
            
            if not os.path.exists(file_path):
                self.logger.error(f"回滚文件不存在: {rollback_file}")
                success = False
                break
            
            try:
                # 读取回滚SQL
                with open(file_path, 'r', encoding='utf-8') as f:
                    sql = f.read()
                
                # 执行回滚
                self.db_manager.execute_update(sql)
                
                # 删除迁移记录
                delete_sql = """
                DELETE FROM schema_migrations 
                WHERE version = %s
                """
                self.db_manager.execute_update(delete_sql, (version,))
                
                self.logger.info(f"迁移 {version} 回滚成功")
                
            except Exception as e:
                self.logger.error(f"迁移 {version} 回滚失败: {e}")
                success = False
                break
        
        return success

def migrate_database():
    """执行数据库迁移"""
    migrator = DatabaseMigrator()
    return migrator.migrate()

def rollback_database(steps: int = 1):
    """回滚数据库迁移
    
    Args:
        steps: 回滚步数
    """
    migrator = DatabaseMigrator()
    return migrator.rollback(steps)

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