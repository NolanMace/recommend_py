#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推荐系统启动脚本
"""
import os
import sys
import logging
import logging.handlers
from datetime import datetime
from recommender.recommender import get_recommender
import time
import signal
import atexit
import json

# 创建日志目录结构
LOG_DIR = 'logs'
LOG_DIRS = {
    'info': os.path.join(LOG_DIR, 'info'),
    'error': os.path.join(LOG_DIR, 'error'),
    'debug': os.path.join(LOG_DIR, 'debug')
}

for dir_path in LOG_DIRS.values():
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

class JsonFormatter(logging.Formatter):
    """JSON格式的日志格式化器"""
    def format(self, record):
        log_obj = {
            "time": self.formatTime(record),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno
        }
        
        if record.exc_info:
            log_obj['exc_info'] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj, ensure_ascii=False)

def setup_logging():
    """配置日志系统"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 日志格式
    standard_formatter = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s [%(module)s:%(lineno)d] - %(message)s'
    )
    
    json_formatter = JsonFormatter()
    
    # INFO级别日志
    info_handler = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIRS['info'], f'recommender_info_{datetime.now().strftime("%Y%m%d")}.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=30,
        encoding='utf-8'
    )
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(standard_formatter)
    logger.addHandler(info_handler)
    
    # ERROR级别日志
    error_handler = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIRS['error'], f'recommender_error_{datetime.now().strftime("%Y%m%d")}.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=30,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(json_formatter)  # 错误日志使用JSON格式
    logger.addHandler(error_handler)
    
    # DEBUG级别日志
    debug_handler = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIRS['debug'], f'recommender_debug_{datetime.now().strftime("%Y%m%d")}.log'),
        maxBytes=20*1024*1024,  # 20MB
        backupCount=10,
        encoding='utf-8'
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(standard_formatter)
    logger.addHandler(debug_handler)
    
    # 控制台输出（开发环境使用）
    if os.environ.get('ENV') != 'production':
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(standard_formatter)
        logger.addHandler(console_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    return logger

# 创建PID文件
def create_pid_file():
    pid = str(os.getpid())
    pid_file = "recommender.pid"
    
    with open(pid_file, "w") as f:
        f.write(pid)
    
    def cleanup():
        try:
            os.unlink(pid_file)
        except:
            pass
    
    atexit.register(cleanup)

def signal_handler(signum, frame):
    """信号处理函数"""
    logger = logging.getLogger('recommender')
    logger.info(f"收到信号 {signum}，准备退出...")
    sys.exit(0)

def main():
    # 设置日志
    logger = setup_logging()
    logger.info("推荐系统启动中...")
    logger.debug("开始初始化系统组件...")
    
    # 创建PID文件
    create_pid_file()
    
    # 注册信号处理
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # 获取推荐器实例
        recommender = get_recommender()
        logger.debug("推荐器实例初始化完成")
        
        # 设置定时任务间隔（秒）
        HOT_TOPICS_INTERVAL = 300  # 5分钟
        RECOMMENDATIONS_INTERVAL = 3600  # 1小时
        
        last_hot_topics_time = 0
        last_recommendations_time = 0
        
        logger.info("推荐系统启动完成，开始运行...")
        
        # 主循环
        while True:
            current_time = time.time()
            
            # 生成热点话题
            if current_time - last_hot_topics_time >= HOT_TOPICS_INTERVAL:
                try:
                    logger.info("开始生成热点话题...")
                    recommender.generate_hot_topics()
                    last_hot_topics_time = current_time
                    logger.debug(f"热点话题生成完成，下次生成时间: {datetime.fromtimestamp(last_hot_topics_time + HOT_TOPICS_INTERVAL)}")
                except Exception as e:
                    logger.error(f"生成热点话题失败: {str(e)}", exc_info=True)
            
            # 批量生成推荐
            if current_time - last_recommendations_time >= RECOMMENDATIONS_INTERVAL:
                try:
                    logger.info("开始批量生成推荐...")
                    recommender.batch_generate_recommendations()
                    last_recommendations_time = current_time
                    logger.debug(f"推荐生成完成，下次生成时间: {datetime.fromtimestamp(last_recommendations_time + RECOMMENDATIONS_INTERVAL)}")
                except Exception as e:
                    logger.error(f"批量生成推荐失败: {str(e)}", exc_info=True)
            
            # 休眠一段时间
            time.sleep(60)  # 每分钟检查一次
            
    except Exception as e:
        logger.error(f"推荐系统运行异常: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 