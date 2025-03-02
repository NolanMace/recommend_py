import logging
import time
import threading
import schedule
import queue
from typing import Dict, List, Callable, Any, Optional, Union
from datetime import datetime, timedelta
import traceback
import json
import os
from dataclasses import dataclass, field, asdict
import uuid

@dataclass
class TaskStatus:
    """任务状态类"""
    id: str  # 任务ID
    name: str  # 任务名称
    scheduled_time: datetime  # 计划执行时间
    status: str = "pending"  # 状态：pending, running, completed, failed
    start_time: Optional[datetime] = None  # 开始时间
    end_time: Optional[datetime] = None  # 结束时间
    duration: float = 0.0  # 执行时长（秒）
    result: Any = None  # 执行结果
    error: Optional[str] = None  # 错误信息
    retries: int = 0  # 重试次数
    max_retries: int = 3  # 最大重试次数
    priority: int = 0  # 优先级（越高越优先）

    def to_dict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        # 将日期时间字段转换为ISO格式字符串
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

    def mark_started(self) -> None:
        """标记任务开始"""
        self.status = "running"
        self.start_time = datetime.now()

    def mark_completed(self, result: Any = None) -> None:
        """标记任务完成"""
        self.status = "completed"
        self.end_time = datetime.now()
        self.result = result
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()

    def mark_failed(self, error: str) -> None:
        """标记任务失败"""
        self.status = "failed"
        self.end_time = datetime.now()
        self.error = error
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()

    def can_retry(self) -> bool:
        """检查是否可以重试"""
        return self.status == "failed" and self.retries < self.max_retries

    def increment_retry(self) -> None:
        """增加重试次数"""
        self.retries += 1
        self.status = "pending"
        self.start_time = None
        self.end_time = None
        self.duration = 0.0
        self.error = None


class PriorityTaskQueue:
    """优先级任务队列"""
    def __init__(self):
        # 优先级队列，值越小优先级越高
        self.queue = queue.PriorityQueue()
        self.lock = threading.Lock()
        
    def put(self, task_status: TaskStatus, priority: int = None) -> None:
        """向队列添加任务
        
        Args:
            task_status: 任务状态对象
            priority: 优先级，不指定则使用任务自带优先级
        """
        # 使用负值使较大的优先级值具有较高的优先级
        if priority is not None:
            task_priority = -priority
        else:
            task_priority = -task_status.priority
            
        with self.lock:
            self.queue.put((task_priority, task_status))
    
    def get(self) -> Optional[TaskStatus]:
        """从队列获取最高优先级的任务"""
        try:
            _, task_status = self.queue.get(block=False)
            return task_status
        except queue.Empty:
            return None
    
    def empty(self) -> bool:
        """检查队列是否为空"""
        return self.queue.empty()
    
    def qsize(self) -> int:
        """获取队列大小"""
        return self.queue.qsize()
    
    def task_done(self) -> None:
        """标记任务完成"""
        self.queue.task_done()


class TaskScheduler:
    """任务调度器"""
    def __init__(self, num_workers: int = 2, log_file: str = None):
        self.logger = logging.getLogger("task_scheduler")
        self.task_queue = PriorityTaskQueue()
        self.workers = []
        self.num_workers = num_workers
        self.running = False
        self.task_history = {}  # 任务历史记录 {task_id: TaskStatus}
        self.history_lock = threading.Lock()
        self.max_history_size = 1000  # 最大历史记录数
        self.log_file = log_file
        
        # 创建定时任务调度器
        self.scheduler = schedule.Scheduler()
        self.scheduler_thread = None
        
    def start(self) -> None:
        """启动调度器"""
        if self.running:
            self.logger.warning("调度器已经在运行")
            return
            
        self.running = True
        
        # 启动工作线程
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"TaskWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            
        # 启动调度器线程
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="SchedulerThread",
            daemon=True
        )
        self.scheduler_thread.start()
        
        self.logger.info(f"任务调度器已启动，工作线程数: {self.num_workers}")
        
    def stop(self) -> None:
        """停止调度器"""
        if not self.running:
            return
            
        self.running = False
        
        # 等待工作线程结束
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=5.0)
                
        # 等待调度器线程结束
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)
            
        self.logger.info("任务调度器已停止")
        
    def add_task(self, task_func: Callable, task_name: str, *args,
                priority: int = 0, max_retries: int = 3, **kwargs) -> str:
        """添加任务到队列
        
        Args:
            task_func: 任务函数
            task_name: 任务名称
            args: 位置参数
            priority: 优先级，越高越优先执行
            max_retries: 最大重试次数
            kwargs: 关键字参数
            
        Returns:
            str: 任务ID
        """
        task_id = str(uuid.uuid4())
        
        # 创建任务状态对象
        task_status = TaskStatus(
            id=task_id,
            name=task_name,
            scheduled_time=datetime.now(),
            priority=priority,
            max_retries=max_retries
        )
        
        # 添加到历史记录
        with self.history_lock:
            self.task_history[task_id] = task_status
            
            # 如果历史记录过多，清理旧记录
            if len(self.task_history) > self.max_history_size:
                self._cleanup_history()
                
        # 包装任务函数
        def task_wrapper():
            return task_func(*args, **kwargs)
            
        # 添加到队列
        self.task_queue.put(
            (task_id, task_name, task_wrapper, task_status),
            priority=priority
        )
        
        self.logger.debug(f"任务已添加: {task_name}, ID: {task_id}, 优先级: {priority}")
        return task_id
        
    def schedule_task(self, task_func: Callable, task_name: str, schedule_type: str,
                     schedule_param: Union[str, int], *args, priority: int = 0,
                     max_retries: int = 3, **kwargs) -> str:
        """定时调度任务
        
        Args:
            task_func: 任务函数
            task_name: 任务名称
            schedule_type: 调度类型（'interval', 'daily', 'weekly', 'monday', etc.）
            schedule_param: 调度参数（interval的秒数或每日时间如'10:30'）
            args: 位置参数
            priority: 优先级
            max_retries: 最大重试次数
            kwargs: 关键字参数
            
        Returns:
            str: 任务ID
        """
        task_id = str(uuid.uuid4())
        
        # 创建一个闭包函数执行实际任务并添加到队列
        def scheduled_job():
            # 创建任务状态对象
            task_status = TaskStatus(
                id=task_id,
                name=task_name,
                scheduled_time=datetime.now(),
                priority=priority,
                max_retries=max_retries
            )
            
            # 添加到历史记录
            with self.history_lock:
                self.task_history[task_id] = task_status
                
                # 如果历史记录过多，清理旧记录
                if len(self.task_history) > self.max_history_size:
                    self._cleanup_history()
                    
            # 包装任务函数
            def task_wrapper():
                return task_func(*args, **kwargs)
                
            # 添加到队列
            self.task_queue.put(
                (task_id, task_name, task_wrapper, task_status),
                priority=priority
            )
            
            self.logger.debug(f"计划任务已触发: {task_name}, ID: {task_id}")
            return schedule.CancelJob  # 如果是重复任务，返回None
            
        # 根据调度类型配置任务
        job = None
        
        if schedule_type == 'interval':
            # schedule_param为秒数
            seconds = int(schedule_param)
            job = self.scheduler.every(seconds).seconds.do(scheduled_job)
        elif schedule_type == 'minutes':
            minutes = int(schedule_param)
            job = self.scheduler.every(minutes).minutes.do(scheduled_job)
        elif schedule_type == 'hourly':
            # schedule_param可以是具体分钟数
            if isinstance(schedule_param, int) or schedule_param.isdigit():
                minutes = int(schedule_param)
                job = self.scheduler.every().hour.at(f":{minutes}").do(scheduled_job)
            else:
                job = self.scheduler.every().hour.do(scheduled_job)
        elif schedule_type == 'daily':
            # schedule_param为时间字符串，如'10:30'
            job = self.scheduler.every().day.at(schedule_param).do(scheduled_job)
        elif schedule_type == 'weekly':
            # schedule_param为时间字符串
            job = self.scheduler.every().week.at(schedule_param).do(scheduled_job)
        elif schedule_type in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
            # 获取对应的调度器方法
            day_method = getattr(self.scheduler.every(), schedule_type)
            # schedule_param为时间字符串
            job = day_method.at(schedule_param).do(scheduled_job)
        else:
            self.logger.error(f"未知的调度类型: {schedule_type}")
            return None
            
        self.logger.info(f"定时任务已设置: {task_name}, 类型: {schedule_type}, 参数: {schedule_param}")
        return task_id
        
    def execute_task_now(self, task_func: Callable, task_name: str, *args, **kwargs) -> Any:
        """立即执行任务（阻塞方式）
        
        Args:
            task_func: 任务函数
            task_name: 任务名称
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            Any: 任务执行结果
        """
        self.logger.info(f"立即执行任务: {task_name}")
        
        task_id = str(uuid.uuid4())
        
        # 创建任务状态对象
        task_status = TaskStatus(
            id=task_id,
            name=task_name,
            scheduled_time=datetime.now()
        )
        
        # 添加到历史记录
        with self.history_lock:
            self.task_history[task_id] = task_status
            
        # 执行任务
        task_status.mark_started()
        
        try:
            result = task_func(*args, **kwargs)
            task_status.mark_completed(result)
            return result
        except Exception as e:
            error_trace = traceback.format_exc()
            task_status.mark_failed(str(e))
            self.logger.error(f"任务执行失败: {task_name}, 错误: {e}\n{error_trace}")
            return None
            
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[Dict]: 任务状态字典或None
        """
        with self.history_lock:
            if task_id in self.task_history:
                return self.task_history[task_id].to_dict()
        return None
        
    def get_all_tasks(self) -> List[Dict]:
        """获取所有任务状态
        
        Returns:
            List[Dict]: 任务状态字典列表
        """
        with self.history_lock:
            return [task.to_dict() for task in self.task_history.values()]
            
    def get_recent_tasks(self, limit: int = 10) -> List[Dict]:
        """获取最近的任务
        
        Args:
            limit: 返回的任务数量
            
        Returns:
            List[Dict]: 任务状态字典列表
        """
        with self.history_lock:
            # 按照调度时间排序，最近的在前
            sorted_tasks = sorted(
                self.task_history.values(),
                key=lambda x: x.scheduled_time if x.scheduled_time else datetime.min,
                reverse=True
            )
            return [task.to_dict() for task in sorted_tasks[:limit]]
            
    def _worker_loop(self) -> None:
        """工作线程循环"""
        thread_name = threading.current_thread().name
        self.logger.debug(f"工作线程 {thread_name} 已启动")
        
        while self.running:
            try:
                # 获取任务，不阻塞
                task = self.task_queue.get()
                
                if task is None:
                    # 没有任务，休息一下
                    time.sleep(0.1)
                    continue
                    
                task_id, task_name, task_func, task_status = task
                
                # 执行任务
                self._execute_task(task_id, task_name, task_func, task_status)
                
                # 标记任务完成
                self.task_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"工作线程 {thread_name} 错误: {e}")
                time.sleep(1)  # 避免因错误导致的CPU占用过高
                
        self.logger.debug(f"工作线程 {thread_name} 已停止")
        
    def _scheduler_loop(self) -> None:
        """调度器线程循环"""
        self.logger.debug("调度器线程已启动")
        
        while self.running:
            try:
                # 运行所有到期的调度任务
                self.scheduler.run_pending()
                
                # 短暂休息
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"调度器线程错误: {e}")
                time.sleep(5)  # 出错后等待一段时间再重试
                
        self.logger.debug("调度器线程已停止")
        
    def _execute_task(self, task_id: str, task_name: str, 
                     task_func: Callable, task_status: TaskStatus) -> None:
        """执行任务"""
        self.logger.info(f"开始执行任务: {task_name}, ID: {task_id}")
        
        # 更新任务状态
        task_status.mark_started()
        
        try:
            # 执行任务
            result = task_func()
            
            # 更新任务状态
            task_status.mark_completed(result)
            
            self.logger.info(f"任务执行成功: {task_name}, ID: {task_id}, 耗时: {task_status.duration:.2f}秒")
            
            # 记录到日志文件
            if self.log_file:
                self._log_task_result(task_status)
                
        except Exception as e:
            error_trace = traceback.format_exc()
            
            # 更新任务状态
            task_status.mark_failed(str(e))
            
            self.logger.error(f"任务执行失败: {task_name}, ID: {task_id}, 错误: {e}\n{error_trace}")
            
            # 记录到日志文件
            if self.log_file:
                self._log_task_result(task_status)
                
            # 检查是否需要重试
            if task_status.can_retry():
                task_status.increment_retry()
                self.logger.info(f"重试任务: {task_name}, ID: {task_id}, 重试次数: {task_status.retries}")
                
                # 添加回队列重试
                self.task_queue.put(
                    (task_id, task_name, task_func, task_status),
                    priority=task_status.priority
                )
                
    def _log_task_result(self, task_status: TaskStatus) -> None:
        """记录任务结果到日志文件"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'task': task_status.to_dict()
                }
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"写入任务日志失败: {e}")
            
    def _cleanup_history(self) -> None:
        """清理过期的历史记录"""
        with self.history_lock:
            # 按时间排序，保留最近的记录
            sorted_items = sorted(
                self.task_history.items(),
                key=lambda x: x[1].scheduled_time if x[1].scheduled_time else datetime.min,
                reverse=True
            )
            
            # 保留前max_history_size个记录
            keep_items = sorted_items[:self.max_history_size]
            
            # 更新历史记录
            self.task_history = {k: v for k, v in keep_items}
            
            self.logger.debug(f"历史记录已清理，当前记录数: {len(self.task_history)}")

# 创建全局调度器实例
_scheduler = None

def get_task_scheduler(num_workers: int = 2, log_file: str = None) -> TaskScheduler:
    """获取任务调度器单例"""
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler(num_workers=num_workers, log_file=log_file)
    return _scheduler 