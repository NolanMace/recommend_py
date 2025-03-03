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
        self.config = self.config_manager.get('scheduler')
        self.database_config = self.config_manager.get('database')
        self.hot_topics_config = self.config_manager.get('hot_topics')
        
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
            new_hot_topics = rec.generate_hot_topics(self.hot_topics_config['count'])
            
            # 更新全局热点列表
            with self.hot_topics_lock:
                self.hot_topics = new_hot_topics
                
            self.logger.info(f"成功生成{len(new_hot_topics)}个热点话题")
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
            batch_size = self.database_config['batch_size']
            success_count = rec.batch_generate_recommendations(batch_size)
            self.logger.info(f"成功为{success_count}/{batch_size}个用户生成推荐")
        except Exception as e:
            self.logger.error(f"批量生成推荐失败: {str(e)}")
    
    def cleanup_expired_data(self):
        """清理过期数据"""
        self.logger.info("开始清理过期数据...")
        try:
            # 清理过期的推荐结果
            expired_time = datetime.now() - timedelta(hours=self.database_config['recommendation_expire_hours'])
            sql = "DELETE FROM user_recommendations WHERE expire_time < %s"
            self.db_pool.execute(sql, (expired_time,))
            
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
                self.db_pool.execute(sql, (max_rec,))
            
            # 清理过旧的历史数据
            cleanup_days = self.database_config['cleanup_days']
            old_time = datetime.now() - timedelta(days=cleanup_days)
            
            # 清理热点历史
            sql = "DELETE FROM hot_topics_history WHERE generation_time < %s"
            self.db_pool.execute(sql, (old_time,))
            
            # 清理曝光记录
            sql = "DELETE FROM post_exposures WHERE exposure_time < %s"
            self.db_pool.execute(sql, (old_time,))
            
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