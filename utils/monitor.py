# -*- coding: utf-8 -*-
"""
监控模块：跟踪系统性能和状态
"""
import threading
import time
import json
import psutil
import os
from datetime import datetime, timedelta
import logging
from collections import deque

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('recommend_monitor')

# 性能指标收集器
class PerformanceCollector:
    def __init__(self, history_size=100):
        self.metrics = {
            'cpu_usage': deque(maxlen=history_size),
            'memory_usage': deque(maxlen=history_size),
            'disk_io': deque(maxlen=history_size),
            'db_queries': deque(maxlen=history_size),
            'recommendations_count': deque(maxlen=history_size),
            'cache_hit_ratio': deque(maxlen=history_size),
            'response_times': deque(maxlen=history_size)
        }
        self.lock = threading.RLock()
        self.start_time = datetime.now()
        
        # 计数器
        self.counters = {
            'total_recommendations': 0,
            'failed_recommendations': 0,
            'db_query_count': 0,
            'slow_queries': 0
        }
    
    def record_metric(self, metric_name, value, timestamp=None):
        """记录性能指标"""
        if metric_name not in self.metrics:
            return
            
        if timestamp is None:
            timestamp = datetime.now()
            
        with self.lock:
            self.metrics[metric_name].append({
                'timestamp': timestamp.isoformat(),
                'value': value
            })
    
    def increment_counter(self, counter_name, value=1):
        """增加计数器值"""
        if counter_name not in self.counters:
            return
            
        with self.lock:
            self.counters[counter_name] += value
    
    def get_metrics(self, metric_name=None, last_n=None):
        """获取指标数据"""
        with self.lock:
            if metric_name:
                if metric_name not in self.metrics:
                    return []
                data = list(self.metrics[metric_name])
                if last_n:
                    return data[-last_n:]
                return data
            
            # 返回所有指标的最新值
            result = {}
            for name, values in self.metrics.items():
                if values:
                    result[name] = values[-1]
            return result
    
    def get_counters(self):
        """获取计数器"""
        with self.lock:
            return self.counters.copy()
    
    def get_uptime(self):
        """获取系统运行时间"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def clear_metrics(self):
        """清空所有指标数据"""
        with self.lock:
            for metric in self.metrics:
                self.metrics[metric].clear()
    
    def get_summary(self):
        """获取系统性能摘要"""
        with self.lock:
            uptime = self.get_uptime()
            days, remainder = divmod(uptime, 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            uptime_str = f"{int(days)}天{int(hours)}小时{int(minutes)}分钟{int(seconds)}秒"
            
            summary = {
                "uptime": uptime_str,
                "counters": self.get_counters(),
                "current_metrics": self.get_metrics()
            }
            
            return summary

# 系统资源监控器
class SystemMonitor:
    def __init__(self, collector, interval=60):
        self.collector = collector
        self.interval = interval
        self.running = False
        self.thread = None
        
    def start(self):
        """启动监控线程"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
        logger.info("系统监控已启动，收集间隔: %d秒", self.interval)
    
    def stop(self):
        """停止监控线程"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None
        
        logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                # 收集CPU使用率
                cpu_percent = psutil.cpu_percent(interval=1)
                self.collector.record_metric('cpu_usage', cpu_percent)
                
                # 收集内存使用情况
                memory = psutil.virtual_memory()
                self.collector.record_metric('memory_usage', {
                    'percent': memory.percent,
                    'used_mb': memory.used / (1024 * 1024),
                    'total_mb': memory.total / (1024 * 1024)
                })
                
                # 收集磁盘I/O
                disk_io = psutil.disk_io_counters()
                self.collector.record_metric('disk_io', {
                    'read_mb': disk_io.read_bytes / (1024 * 1024),
                    'write_mb': disk_io.write_bytes / (1024 * 1024)
                })
                
                # 记录进程信息
                process = psutil.Process(os.getpid())
                self.collector.record_metric('process_info', {
                    'cpu_percent': process.cpu_percent(interval=0.1),
                    'memory_percent': process.memory_percent(),
                    'threads': process.num_threads()
                })
                
                # 检查是否有告警条件
                self._check_alerts()
                
            except Exception as e:
                logger.error("监控数据收集失败: %s", str(e))
            
            # 等待下一个收集间隔
            time.sleep(self.interval)
    
    def _check_alerts(self):
        """检查是否需要发出系统告警"""
        # CPU使用率告警
        cpu_metrics = self.collector.get_metrics('cpu_usage', last_n=3)
        if cpu_metrics and len(cpu_metrics) == 3:
            cpu_values = [m['value'] for m in cpu_metrics]
            avg_cpu = sum(cpu_values) / len(cpu_values)
            if avg_cpu > 85:
                logger.warning("⚠️ 高CPU使用率告警: %.2f%%", avg_cpu)
        
        # 内存使用告警
        memory_metrics = self.collector.get_metrics('memory_usage', last_n=1)
        if memory_metrics:
            memory_percent = memory_metrics[0]['value']['percent']
            if memory_percent > 90:
                logger.warning("⚠️ 高内存使用率告警: %.2f%%", memory_percent)

# 性能追踪装饰器
def track_performance(name, collector):
    """追踪函数执行时间的装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = (time.time() - start_time) * 1000  # 转为毫秒
            
            # 记录响应时间
            collector.record_metric('response_times', {
                'name': name,
                'time_ms': elapsed_time
            })
            
            # 记录慢查询
            if elapsed_time > 1000:  # 超过1秒视为慢操作
                collector.increment_counter('slow_queries')
                logger.warning("慢操作警告: %s 耗时 %.2f 毫秒", name, elapsed_time)
                
            return result
        return wrapper
    return decorator

# 创建全局监控收集器
performance_collector = PerformanceCollector()

# 创建系统监控器
system_monitor = SystemMonitor(performance_collector)

# 启动监控的函数
def start_monitoring():
    """启动系统监控"""
    system_monitor.start()
    logger.info("推荐系统监控已启动")

# 提供HTTP接口查看监控数据
def get_monitoring_data():
    """获取监控数据的JSON"""
    data = {
        'summary': performance_collector.get_summary(),
        'metrics': {
            'cpu': list(performance_collector.get_metrics('cpu_usage')),
            'memory': list(performance_collector.get_metrics('memory_usage')),
            'response_times': list(performance_collector.get_metrics('response_times'))
        }
    }
    return json.dumps(data, indent=2)

# 导出监控指标到日志文件
def export_metrics_log(log_file):
    """将监控指标导出到日志文件"""
    with open(log_file, 'w') as f:
        json.dump(performance_collector.get_summary(), f, indent=2)
    logger.info("监控指标已导出到: %s", log_file)

# 记录函数耗时的上下文管理器
class TimerContext:
    def __init__(self, name, collector=performance_collector):
        self.name = name
        self.collector = collector
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed_time = (time.time() - self.start_time) * 1000  # 毫秒
            self.collector.record_metric('response_times', {
                'name': self.name,
                'time_ms': elapsed_time
            })
            
            if exc_type:
                logger.error("操作 %s 失败，耗时 %.2f 毫秒: %s", 
                            self.name, elapsed_time, str(exc_val))
            elif elapsed_time > 1000:
                logger.warning("操作 %s 耗时较长: %.2f 毫秒", self.name, elapsed_time) 