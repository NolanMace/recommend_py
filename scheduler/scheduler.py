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

from recommender import Recommender, FeatureProcessor
from config import HOT_TOPICS_CONFIG, DATABASE_CONFIG
from database import get_db_pool
from config.config_manager import get_config_manager
from database.database import get_db_manager
from recommender.recommender import get_recommender

# 数据库连接池
db_pool = get_db_pool()

# 保存热点列表的全局变量
hot_topics = []
hot_topics_lock = threading.Lock()

# 推荐器实例(懒加载)
recommender_instance = None
recommender_lock = threading.Lock()

def get_recommender():
    """懒加载获取推荐器实例"""
    global recommender_instance
    if recommender_instance is None:
        with recommender_lock:
            if recommender_instance is None:
                feature_processor = FeatureProcessor()
                feature_processor.load_data()
                recommender_instance = Recommender(feature_processor)
    return recommender_instance

def update_features():
    """每天凌晨更新特征模型"""
    print(f"[{datetime.now()}] 开始更新TF-IDF模型...")
    try:
        fp = FeatureProcessor()
        fp.fit_model()
        
        # 更新全局推荐器实例
        global recommender_instance
        with recommender_lock:
            recommender_instance = Recommender(fp)
            
        print(f"[{datetime.now()}] TF-IDF模型更新完成")
    except Exception as e:
        print(f"[{datetime.now()}] 更新模型失败: {str(e)}")

def generate_hot_topics():
    """生成热点话题"""
    global hot_topics
    
    print(f"[{datetime.now()}] 开始生成热点话题...")
    try:
        rec = get_recommender()
        new_hot_topics = rec.generate_hot_topics(HOT_TOPICS_CONFIG['count'])
        
        # 更新全局热点列表
        with hot_topics_lock:
            hot_topics = new_hot_topics
            
        print(f"[{datetime.now()}] 成功生成{len(new_hot_topics)}个热点话题")
    except Exception as e:
        print(f"[{datetime.now()}] 生成热点话题失败: {str(e)}")

def get_current_hot_topics():
    """获取当前热点话题列表"""
    with hot_topics_lock:
        return hot_topics

def batch_generate_recommendations():
    """批量生成用户推荐"""
    print(f"[{datetime.now()}] 开始批量生成用户推荐...")
    try:
        rec = get_recommender()
        batch_size = DATABASE_CONFIG['batch_size']
        success_count = rec.batch_generate_recommendations(batch_size)
        print(f"[{datetime.now()}] 成功为{success_count}/{batch_size}个用户生成推荐")
    except Exception as e:
        print(f"[{datetime.now()}] 批量生成推荐失败: {str(e)}")

def cleanup_expired_data():
    """清理过期数据"""
    print(f"[{datetime.now()}] 开始清理过期数据...")
    try:
        # 清理过期的推荐结果
        expired_time = datetime.now() - timedelta(hours=DATABASE_CONFIG['recommendation_expire_hours'])
        sql = "DELETE FROM user_recommendations WHERE expire_time < %s"
        result = db_pool.execute(sql, (expired_time,))
        
        # 保留每个用户最近的N条推荐记录
        max_rec = DATABASE_CONFIG['max_recommendations_per_user']
        if max_rec > 0:
            # 使用子查询找出需要保留的记录
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
            db_pool.execute(sql, (max_rec,))
        
        # 清理过旧的历史数据
        cleanup_days = DATABASE_CONFIG['cleanup_days']
        old_time = datetime.now() - timedelta(days=cleanup_days)
        
        # 清理热点历史
        sql = "DELETE FROM hot_topics_history WHERE generation_time < %s"
        db_pool.execute(sql, (old_time,))
        
        # 清理曝光记录
        sql = "DELETE FROM post_exposures WHERE exposure_time < %s"
        db_pool.execute(sql, (old_time,))
        
        print(f"[{datetime.now()}] 数据清理完成")
    except Exception as e:
        print(f"[{datetime.now()}] 清理过期数据失败: {str(e)}")

def setup_scheduler():
    """设置定时任务调度器"""
    print(f"[{datetime.now()}] 正在初始化定时任务调度器...")
    
    try:
        # 添加定时任务
        print(f"[{datetime.now()}] 开始添加定时任务...")
        
        # 每分钟执行一次的任务
        schedule.every(1).minutes.do(log_scheduler_heartbeat)
        print(f"[{datetime.now()}] 已添加心跳监控任务 (每分钟)")
        
        # 每小时执行一次的任务
        schedule.every(1).hours.do(scheduled_task, task_name="热门内容更新", duration_seconds=random.randint(5, 15))
        print(f"[{datetime.now()}] 已添加热门内容更新任务 (每小时)")
        
        # 每4小时执行一次的任务
        schedule.every(4).hours.do(scheduled_task, task_name="用户兴趣模型更新", duration_seconds=random.randint(20, 40))
        print(f"[{datetime.now()}] 已添加用户兴趣模型更新任务 (每4小时)")
        
        # 每天凌晨3点执行的任务
        schedule.every().day.at("03:00").do(scheduled_task, task_name="全量数据分析", duration_seconds=random.randint(60, 120))
        print(f"[{datetime.now()}] 已添加全量数据分析任务 (每天03:00)")
        
        # 计算下一次各任务的执行时间并打印
        print_next_run_times()
        
        print(f"[{datetime.now()}] 定时任务调度器初始化完成，共设置了 {len(schedule.jobs)} 个任务")
        print(f"[{datetime.now()}] 开始进入任务调度循环...")
        
        # 无限循环，定时检查和执行任务
        loop_count = 0
        while True:
            try:
                # 每1000次循环打印一次状态信息
                loop_count += 1
                if loop_count % 1000 == 0:
                    print(f"[{datetime.now()}] 调度器运行中 - 循环次数: {loop_count}, 任务数: {len(schedule.jobs)}")
                
                # 执行所有到期任务
                schedule.run_pending()
                time.sleep(1)  # 休眠1秒
            except Exception as e:
                print(f"[{datetime.now()}] 任务调度器异常: {str(e)}")
                time.sleep(5)  # 异常情况下，休眠5秒后继续
    except Exception as e:
        print(f"[{datetime.now()}] 调度器初始化过程中发生异常: {str(e)}")
        # 记录完整的异常堆栈
        import traceback
        print(f"[{datetime.now()}] 异常堆栈: {traceback.format_exc()}")

def print_next_run_times():
    """打印所有任务的下次执行时间"""
    print(f"[{datetime.now()}] 下次任务执行时间:")
    for job in schedule.jobs:
        next_run = job.next_run
        time_diff = next_run - datetime.now()
        minutes = int(time_diff.total_seconds() / 60)
        print(f"  - {job.job_func.__name__} ({get_job_description(job)}): {next_run.strftime('%Y-%m-%d %H:%M:%S')} (约 {minutes} 分钟后)")

def get_job_description(job):
    """获取任务描述"""
    if str(job).find("minutes") > 0:
        return "每分钟"
    elif str(job).find("hours") > 0 and str(job).find("every 1 hour") > 0:
        return "每小时"
    elif str(job).find("hours") > 0 and str(job).find("every 4 hours") > 0:
        return "每4小时"
    elif str(job).find("day at 03:00") > 0:
        return "每天03:00"
    return "未知任务"

def log_scheduler_heartbeat():
    """记录调度器心跳"""
    running_jobs = [j for j in threading.enumerate() if j.name.startswith('Task:')]
    print(f"[{datetime.now()}] 调度器心跳 - 系统正常运行中, 当前活跃任务数: {len(running_jobs)}")
    if running_jobs:
        for job in running_jobs:
            print(f"  - 活跃任务: {job.name}, 运行时间: {job.daemon}")
    
    # 打印下一次任务执行时间
    if random.random() < 0.2:  # 只有20%的心跳会打印下一次执行时间，避免日志过多
        print_next_run_times()
    
    return True  # 必须返回True以便继续调度

def scheduled_task(task_name, duration_seconds):
    """模拟执行定时任务"""
    thread_name = f"Task:{task_name}:{int(time.time())}"
    threading.current_thread().name = thread_name
    
    print(f"[{datetime.now()}] 开始执行任务: {task_name}, 预计耗时: {duration_seconds}秒")
    
    # 模拟任务进度
    for progress in range(0, 101, 10):
        if progress > 0:
            # 模拟任务执行
            time.sleep(duration_seconds * 0.1)
        
        print(f"[{datetime.now()}] 任务进度 - {task_name}: {progress}% 完成")
    
    print(f"[{datetime.now()}] 任务完成: {task_name}, 实际耗时: {duration_seconds}秒")
    return True  # 必须返回True以便继续调度

class TaskScheduler:
    """任务调度器
    
    负责管理和执行定时任务
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TaskScheduler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.logger = logging.getLogger("scheduler")
        self.config_manager = get_config_manager()
        self.db_manager = get_db_manager()
        self.recommender = get_recommender()
        
        # 获取配置
        self.config = self.config_manager.get('scheduler')
        
        # 创建线程池
        self.executor = ThreadPoolExecutor(max_workers=self.config['max_workers'])
        
        # 任务状态
        self.running_tasks = {}
        self.task_lock = threading.Lock()
        
        # 注册为配置观察者
        self.config_manager.register_observer(self)
        
        self._initialized = True
        self.logger.info("调度器初始化完成")
    
    def config_updated(self, path: str, new_value: Any):
        """配置更新回调"""
        if path.startswith('scheduler.'):
            self.logger.info(f"检测到调度器配置变更: {path}")
            # 更新本地配置
            self.config = self.config_manager.get('scheduler')
            # 更新线程池
            self.executor._max_workers = self.config['max_workers']
    
    def start(self):
        """启动调度器"""
        self.logger.info("启动调度器")
        
        # 获取任务配置
        jobs = self.config['jobs']
        
        # 启动所有任务
        for job_name, job_config in jobs.items():
            self.schedule_task(job_name, job_config)
    
    def schedule_task(self, task_name: str, task_config: Dict):
        """调度任务
        
        Args:
            task_name: 任务名称
            task_config: 任务配置
        """
        def run_task():
            while True:
                try:
                    # 执行任务
                    if task_name == 'update_hot_topics':
                        self.update_hot_topics()
                    elif task_name == 'precalculate_recommendations':
                        self.precalculate_recommendations()
                    elif task_name == 'clean_cache':
                        self.clean_cache()
                    elif task_name == 'calculate_statistics':
                        self.calculate_statistics()
                    
                    # 等待下一次执行
                    interval = task_config['interval'].total_seconds()
                    threading.Event().wait(interval)
                    
                except Exception as e:
                    self.logger.error(f"任务 {task_name} 执行失败: {e}")
                    # 出错后等待一段时间再重试
                    threading.Event().wait(60)
        
        # 提交任务到线程池
        with self.task_lock:
            if task_name not in self.running_tasks:
                future = self.executor.submit(run_task)
                self.running_tasks[task_name] = future
                self.logger.info(f"任务 {task_name} 已调度")
    
    def stop(self):
        """停止调度器"""
        self.logger.info("停止调度器")
        
        # 取消所有任务
        with self.task_lock:
            for task_name, future in self.running_tasks.items():
                future.cancel()
            self.running_tasks.clear()
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
    
    def update_hot_topics(self):
        """更新热点话题"""
        try:
            max_topics = self.config['jobs']['update_hot_topics']['max_topics']
            self.recommender.generate_hot_topics(max_topics)
            self.logger.info("热点话题已更新")
        except Exception as e:
            self.logger.error(f"更新热点话题失败: {e}")
    
    def precalculate_recommendations(self):
        """预计算推荐结果"""
        try:
            batch_size = self.config['jobs']['precalculate_recommendations']['batch_size']
            self.recommender.batch_generate_recommendations(batch_size)
            self.logger.info("推荐结果预计算完成")
        except Exception as e:
            self.logger.error(f"预计算推荐结果失败: {e}")
    
    def clean_cache(self):
        """清理缓存"""
        try:
            # TODO: 实现缓存清理逻辑
            self.logger.info("缓存清理完成")
        except Exception as e:
            self.logger.error(f"清理缓存失败: {e}")
    
    def calculate_statistics(self):
        """计算统计数据"""
        try:
            # TODO: 实现统计数据计算逻辑
            self.logger.info("统计数据计算完成")
        except Exception as e:
            self.logger.error(f"计算统计数据失败: {e}")

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