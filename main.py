#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统启动入口
"""
import threading
import argparse
import os
import signal
import sys
import time
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

from scheduler.scheduler import TaskScheduler, scheduled_task
from database import get_db_pool

# 全局退出标志
should_exit = False

def init_logging():
    """初始化日志系统"""
    # 获取项目根目录下的logs目录
    log_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    
    # 创建日志目录
    log_dirs = ['info', 'error', 'debug']
    for dir_name in log_dirs:
        dir_path = os.path.join(log_base_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    
    # 配置根日志器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 日志格式
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 按天切割的INFO日志文件
    info_handler = TimedRotatingFileHandler(
        os.path.join(log_base_dir, 'info', 'recommend.log'),
        when='midnight',
        interval=1,
        backupCount=30,  # 保留30天的日志
        encoding='utf-8'
    )
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    
    # 错误日志文件
    error_handler = TimedRotatingFileHandler(
        os.path.join(log_base_dir, 'error', 'error.log'),
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    
    # 控制台输出
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console)
    
    return logger

def init_tables():
    """初始化数据库表结构"""
    try:
        logging.info("开始初始化数据库表结构...")
        db_pool = get_db_pool()
        
        # 读取Schema脚本
        script_path = os.path.join(os.path.dirname(__file__), 'schema_update.sql')
        if os.path.exists(script_path):
            logging.info(f"找到数据库脚本文件: {script_path}")
            with open(script_path, 'r') as f:
                script = f.read()
            
            logging.info("开始执行数据库脚本...")
            # 使用新的execute_script方法执行SQL脚本
            result = db_pool.execute_script(script, batch_size=20)
            
            logging.info(f"数据库表结构初始化完成，总计 {result['total']} 条SQL语句")
            logging.info(f"成功: {result['success']}，失败: {result['error']}，总耗时: {result['duration']:.2f}秒")
        else:
            logging.warning("警告: 未找到数据库脚本文件")
    except Exception as e:
        logging.error(f"初始化数据库表结构失败: {str(e)}", exc_info=True)

def handle_signal(signum, frame):
    """处理退出信号"""
    global should_exit
    logging.info("\n收到退出信号，正在安全退出...")
    should_exit = True

def run_all_tasks():
    """立即运行所有主要任务"""
    logging.info("开始执行所有主要任务...")
    
    # 执行热门内容更新任务
    logging.info("1/3 开始执行热门内容更新任务")
    scheduled_task("热门内容更新", 15)
    
    # 执行用户兴趣模型更新任务
    logging.info("2/3 开始执行用户兴趣模型更新任务")
    scheduled_task("用户兴趣模型更新", 40)
    
    # 执行全量数据分析任务
    logging.info("3/3 开始执行全量数据分析任务")
    scheduled_task("全量数据分析", 120)
    
    logging.info("所有主要任务执行完成")

def main():
    """主程序入口"""
    # 初始化日志系统
    logger = init_logging()
    start_time = time.time()
    
    logger.info("推荐系统启动中...")
    
    parser = argparse.ArgumentParser(description='推荐系统服务')
    parser.add_argument('--debug', action='store_true', help='是否启用调试模式')
    parser.add_argument('--init-db', action='store_true', help='是否初始化数据库表结构(默认不初始化)')
    parser.add_argument('--no-scheduler', action='store_true', help='是否禁用调度任务服务')
    parser.add_argument('--run-tasks', action='store_true', help='立即执行三个主要任务')
    args = parser.parse_args()
    
    logger.info(f"启动参数: {args}")
    logger.info(f"运行环境: Python {sys.version}, OS: {os.name}")
    
    try:
        # 初始化数据库连接池
        logger.info("正在初始化数据库连接池...")
        db_pool = get_db_pool()
        logger.info("数据库连接池初始化完成")
        
        # 初始化数据库表结构
        if args.init_db:
            init_tables()
        else:
            logger.info("跳过数据库表结构初始化，如需初始化请使用 --init-db 参数")
        
        # 注册信号处理
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        logger.info("信号处理函数已注册")
        
        # 启动调度器
        scheduler_thread = None
        if not args.no_scheduler:
            logger.info("启动调度任务服务...")
            scheduler = TaskScheduler()
            scheduler_thread = threading.Thread(target=scheduler.start, daemon=True)
            scheduler_thread.start()
            logger.info("调度任务线程已启动")
        else:
            logger.info("已禁用调度任务服务")
        
        # 如果指定了立即执行任务
        if args.run_tasks:
            logger.info("检测到--run-tasks参数，准备立即执行所有主要任务...")
            run_all_tasks()
        
        startup_time = time.time() - start_time
        logger.info(f"系统启动完成，耗时: {startup_time:.2f}秒")
        logger.info("系统运行中，按Ctrl+C终止...")
        
        # 主循环
        last_status_time = time.time()
        iterations = 0
        while not should_exit:
            iterations += 1
            current_time = time.time()
            
            # 每分钟输出一次状态日志
            if current_time - last_status_time > 60:
                logger.info(f"系统运行正常，已运行: {(current_time - start_time) / 60:.1f}分钟")
                last_status_time = current_time
            
            # 每100次循环输出一次调试日志
            if iterations % 100 == 0:
                logger.debug(f"主循环健康检查 - 迭代次数: {iterations}")
            
            # 如果有调度器线程，检查它是否还活着
            if scheduler_thread:
                scheduler_thread.join(1.0)
            else:
                time.sleep(1.0)
                
    except KeyboardInterrupt:
        logger.info("\n收到键盘中断，正在退出...")
    except Exception as e:
        logger.error(f"系统运行异常: {str(e)}", exc_info=True)
    finally:
        total_runtime = time.time() - start_time
        logger.info(f"系统已安全退出，总运行时间: {total_runtime:.2f}秒")

if __name__ == '__main__':
    main()
