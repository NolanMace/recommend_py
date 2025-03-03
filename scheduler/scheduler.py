# -*- coding: utf-8 -*-
"""
定时更新任务
"""
import time
import schedule
import threading
from datetime import datetime, timedelta
import random
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

from database import get_db_pool
from config.config_manager import get_config_manager

# 避免循环导入
recommender = None
FeatureProcessor = None

def scheduled_task(task_name: str, interval_minutes: int = 0):
    """执行调度任务
    
    Args:
        task_name: 任务名称
        interval_minutes: 任务间隔（分钟）
        
    Returns:
        bool: 任务是否执行成功
    """
    scheduler = get_scheduler()
    logger = logging.getLogger("scheduler")
    
    try:
        logger.info(f"开始执行任务: {task_name}")
        
        if task_name == "热门内容更新":
            scheduler.generate_hot_topics()
        elif task_name == "用户兴趣模型更新":
            scheduler.update_features()
        elif task_name == "全量数据分析":
            scheduler.batch_generate_recommendations()
        else:
            logger.warning(f"未知的任务类型: {task_name}")
            return False
            
        logger.info(f"任务执行完成: {task_name}")
        return True
        
    except Exception as e:
        logger.error(f"任务执行失败 {task_name}: {str(e)}")
        return False

def init_recommender():
    """延迟导入推荐器模块"""
    global recommender, FeatureProcessor
    if recommender is None:
        from recommender.recommender import get_recommender, FeatureProcessor as FP
        recommender = get_recommender
        FeatureProcessor = FP

class TaskScheduler:
    """任务调度器
    
    负责管理和执行定时任务
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TaskScheduler, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.logger = logging.getLogger("scheduler")
        self.config_manager = get_config_manager()
        
        # 获取配置
        self.config = self.config_manager.get('scheduler', {})
        self.database_config = self.config_manager.get('database', {})
        self.hot_topics_config = self.config_manager.get('hot_topics', {})
        
        # 创建线程池
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4))
        
        # 初始化数据库连接池
        self.db_pool = get_db_pool()
        
        # 保存热点列表
        self.hot_topics = []
        self.hot_topics_lock = threading.Lock()
        
        # 推荐器实例(懒加载)
        self.recommender_instance = None
        self.recommender_lock = threading.Lock()
        
        # 任务状态
        self.running_tasks = {}
        self.task_lock = threading.Lock()
        
        # 注册为配置观察者
        self.config_manager.register_observer(self)
        
        self._initialized = True
        self.logger.info("调度器初始化完成")
    
    def get_recommender(self):
        """懒加载获取推荐器实例"""
        if self.recommender_instance is None:
            with self.recommender_lock:
                if self.recommender_instance is None:
                    init_recommender()  # 延迟导入
                    feature_processor = FeatureProcessor()
                    feature_processor.load_data()
                    self.recommender_instance = recommender()
        return self.recommender_instance
    
    def config_updated(self, path: str, new_value: Any):
        """配置更新回调"""
        if path.startswith('scheduler.'):
            self.logger.info(f"检测到调度器配置变更: {path}")
            # 更新本地配置
            self.config = self.config_manager.get('scheduler')
            self.database_config = self.config_manager.get('database')
            self.hot_topics_config = self.config_manager.get('hot_topics')
            # 更新线程池
            self.executor._max_workers = self.config.get('max_workers', 4)
    
    def update_features(self):
        """更新特征模型"""
        self.logger.info("开始更新TF-IDF模型...")
        try:
            fp = FeatureProcessor()
            fp.fit_model()
            
            # 更新全局推荐器实例
            with self.recommender_lock:
                self.recommender_instance = self.get_recommender()
                
            self.logger.info("TF-IDF模型更新完成")
        except Exception as e:
            self.logger.error(f"更新模型失败: {str(e)}")
    
    def generate_hot_topics(self):
        """生成热点话题"""
        self.logger.info("开始生成热点话题...")
        try:
            rec = self.get_recommender()
            # 使用 max_topics 而不是 count
            max_topics = self.hot_topics_config.get('max_topics', 50)
            new_hot_topics = rec.generate_hot_topics()
            
            # 更新全局热点列表
            with self.hot_topics_lock:
                self.hot_topics = new_hot_topics[:max_topics] if new_hot_topics else []
                
            self.logger.info(f"成功生成{len(self.hot_topics)}个热点话题")
        except Exception as e:
            self.logger.error(f"生成热点话题失败: {str(e)}")
    
    def get_current_hot_topics(self):
        """获取当前热点话题列表"""
        with self.hot_topics_lock:
            return self.hot_topics
    
    def batch_generate_recommendations(self):
        """批量生成用户推荐"""
        self.logger.info("开始批量生成用户推荐...")
        try:
            rec = self.get_recommender()
            # 从配置中获取batch_size，如果没有配置则使用默认值1000
            batch_size = self.config.get('batch_size', 1000)
            self.logger.info(f"使用批处理大小: {batch_size}")
            
            # 获取需要处理的用户列表
            sql = """
            SELECT DISTINCT user_id 
            FROM (
                SELECT user_id FROM user_views 
                WHERE created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
                UNION
                SELECT user_id FROM post_likes
                WHERE created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
                UNION
                SELECT user_id FROM post_collects
                WHERE created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
            ) as active_users
            ORDER BY user_id
            LIMIT %s
            """
            
            try:
                users = self.db_pool.execute_query(sql, (batch_size,))
            except Exception as e:
                self.logger.warning(f"查询活跃用户失败: {str(e)}")
                users = []
                
            if not users:
                self.logger.info("没有找到需要处理的用户，任务完成")
                return 0
                
            total_users = len(users)
            success_count = 0
            error_count = 0
            
            self.logger.info(f"开始为 {total_users} 个用户生成推荐...")
            
            for i, user in enumerate(users, 1):
                user_id = user['user_id']
                try:
                    # 为用户生成推荐
                    recommendations = rec.get_recommendations(user_id, page=1, page_size=20)
                    if recommendations:
                        success_count += 1
                    
                    # 每处理100个用户输出一次进度
                    if i % 100 == 0:
                        self.logger.info(f"进度: {i}/{total_users} ({i/total_users*100:.1f}%)")
                        
                except Exception as e:
                    error_count += 1
                    self.logger.error(f"为用户 {user_id} 生成推荐失败: {str(e)}")
                    continue
            
            self.logger.info(f"批量推荐生成完成: 成功 {success_count}/{total_users} 个用户，失败 {error_count} 个")
            return success_count
            
        except Exception as e:
            self.logger.error(f"批量生成推荐失败: {str(e)}")
            return 0
    
    def cleanup_expired_data(self):
        """清理过期数据"""
        self.logger.info("开始清理过期数据...")
        try:
            # 清理过期的推荐结果
            expired_time = datetime.now() - timedelta(hours=self.database_config['recommendation_expire_hours'])
            sql = "DELETE FROM user_recommendations WHERE expire_time < %s"
            self.db_pool.execute_update(sql, {'expire_time': expired_time})
            
            # 保留每个用户最近的N条推荐记录
            max_rec = self.database_config['max_recommendations_per_user']
            if max_rec > 0:
                sql = """
                DELETE FROM user_recommendations 
                WHERE id NOT IN (
                    SELECT id FROM (
                        SELECT id FROM user_recommendations
                        ORDER BY recommendation_time DESC
                        LIMIT %s
                    ) AS latest_recs
                )
                """
                self.db_pool.execute_update(sql, {'limit': max_rec})
            
            # 清理过旧的历史数据
            cleanup_days = self.database_config['cleanup_days']
            old_time = datetime.now() - timedelta(days=cleanup_days)
            
            # 清理热点历史
            sql = "DELETE FROM hot_topics_history WHERE generation_time < %s"
            self.db_pool.execute_update(sql, {'generation_time': old_time})
            
            # 清理曝光记录
            sql = "DELETE FROM post_exposures WHERE exposure_time < %s"
            self.db_pool.execute_update(sql, {'exposure_time': old_time})
            
            self.logger.info("数据清理完成")
        except Exception as e:
            self.logger.error(f"清理过期数据失败: {str(e)}")
    
    def start(self):
        """启动调度器"""
        self.logger.info("启动调度器")
        
        try:
            # 添加定时任务
            self.logger.info("开始添加定时任务...")
            
            # 每分钟执行一次的任务
            schedule.every(1).minutes.do(self.log_heartbeat)
            self.logger.info("已添加心跳监控任务 (每分钟)")
            
            # 每小时执行一次的任务
            schedule.every(1).hours.do(self.generate_hot_topics)
            self.logger.info("已添加热门内容更新任务 (每小时)")
            
            # 每4小时执行一次的任务
            schedule.every(4).hours.do(self.batch_generate_recommendations)
            self.logger.info("已添加用户推荐生成任务 (每4小时)")
            
            # 每天凌晨3点执行的任务
            schedule.every().day.at("03:00").do(self.update_features)
            self.logger.info("已添加特征更新任务 (每天03:00)")
            
            # 每天凌晨4点执行的任务
            schedule.every().day.at("04:00").do(self.cleanup_expired_data)
            self.logger.info("已添加数据清理任务 (每天04:00)")
            
            # 打印下一次执行时间
            self.print_next_run_times()
            
            self.logger.info(f"定时任务调度器初始化完成，共设置了 {len(schedule.jobs)} 个任务")
            self.logger.info("开始进入任务调度循环...")
            
            # 无限循环，定时检查和执行任务
            loop_count = 0
            while True:
                try:
                    # 每1000次循环打印一次状态信息
                    loop_count += 1
                    if loop_count % 1000 == 0:
                        self.logger.info(f"调度器运行中 - 循环次数: {loop_count}, 任务数: {len(schedule.jobs)}")
                    
                    # 执行所有到期任务
                    schedule.run_pending()
                    time.sleep(1)  # 休眠1秒
                except Exception as e:
                    self.logger.error(f"任务调度器异常: {str(e)}")
                    time.sleep(5)  # 异常情况下，休眠5秒后继续
        except Exception as e:
            self.logger.error(f"调度器初始化过程中发生异常: {str(e)}")
            # 记录完整的异常堆栈
            import traceback
            self.logger.error(f"异常堆栈: {traceback.format_exc()}")
    
    def print_next_run_times(self):
        """打印所有任务的下次执行时间"""
        self.logger.info("下次任务执行时间:")
        for job in schedule.jobs:
            next_run = job.next_run
            time_diff = next_run - datetime.now()
            minutes = int(time_diff.total_seconds() / 60)
            self.logger.info(f"  - {job.job_func.__name__} ({self.get_job_description(job)}): "
                           f"{next_run.strftime('%Y-%m-%d %H:%M:%S')} (约 {minutes} 分钟后)")
    
    def get_job_description(self, job):
        """获取任务描述"""
        if str(job).find("minutes") > 0:
            return "每分钟"
        elif str(job).find("hours") > 0 and str(job).find("every 1 hour") > 0:
            return "每小时"
        elif str(job).find("hours") > 0 and str(job).find("every 4 hours") > 0:
            return "每4小时"
        elif str(job).find("day at 03:00") > 0:
            return "每天03:00"
        elif str(job).find("day at 04:00") > 0:
            return "每天04:00"
        return "未知任务"
    
    def log_heartbeat(self):
        """记录调度器心跳"""
        running_jobs = [j for j in threading.enumerate() if j.name.startswith('Task:')]
        self.logger.info(f"调度器心跳 - 系统正常运行中, 当前活跃任务数: {len(running_jobs)}")
        if running_jobs:
            for job in running_jobs:
                self.logger.info(f"  - 活跃任务: {job.name}, 运行时间: {job.daemon}")
        
        # 打印下一次任务执行时间
        if random.random() < 0.2:  # 只有20%的心跳会打印下一次执行时间，避免日志过多
            self.print_next_run_times()
        
        return True  # 必须返回True以便继续调度

# 全局调度器实例
_scheduler = None

def get_scheduler() -> TaskScheduler:
    """获取调度器实例
    
    Returns:
        TaskScheduler: 调度器实例
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler