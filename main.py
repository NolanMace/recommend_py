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
from datetime import datetime

from scheduler.scheduler import TaskScheduler, scheduled_task
from database import get_db_pool

# 全局退出标志
should_exit = False

def init_tables():
    """初始化数据库表结构"""
    try:
        print(f"[{datetime.now()}] 开始初始化数据库表结构...")
        db_pool = get_db_pool()
        
        # 读取Schema脚本
        script_path = os.path.join(os.path.dirname(__file__), 'schema_update.sql')
        if os.path.exists(script_path):
            print(f"[{datetime.now()}] 找到数据库脚本文件: {script_path}")
            with open(script_path, 'r') as f:
                script = f.read()
            
            print(f"[{datetime.now()}] 开始执行数据库脚本...")
            # 使用新的execute_script方法执行SQL脚本
            result = db_pool.execute_script(script, batch_size=20)
            
            print(f"[{datetime.now()}] 数据库表结构初始化完成，总计 {result['total']} 条SQL语句")
            print(f"[{datetime.now()}] 成功: {result['success']}，失败: {result['error']}，总耗时: {result['duration']:.2f}秒")
        else:
            print(f"[{datetime.now()}] 警告: 未找到数据库脚本文件")
    except Exception as e:
        print(f"[{datetime.now()}] 初始化数据库表结构失败: {str(e)}")

def handle_signal(signum, frame):
    """处理退出信号"""
    global should_exit
    print(f"\n[{datetime.now()}] 收到退出信号，正在安全退出...")
    should_exit = True

def run_all_tasks():
    """立即运行所有主要任务"""
    print(f"[{datetime.now()}] 开始执行所有主要任务...")
    
    # 执行热门内容更新任务
    print(f"[{datetime.now()}] 1/3 开始执行热门内容更新任务")
    scheduled_task("热门内容更新", 15)
    
    # 执行用户兴趣模型更新任务
    print(f"[{datetime.now()}] 2/3 开始执行用户兴趣模型更新任务")
    scheduled_task("用户兴趣模型更新", 40)
    
    # 执行全量数据分析任务
    print(f"[{datetime.now()}] 3/3 开始执行全量数据分析任务")
    scheduled_task("全量数据分析", 120)
    
    print(f"[{datetime.now()}] 所有主要任务执行完成")

def main():
    """主程序入口"""
    # 创建日志目录
    log_dirs = ['info', 'error', 'debug']
    for dir_name in log_dirs:
        dir_path = os.path.join('logs', dir_name)
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
                print(f"创建日志目录: {dir_path}")
            except Exception as e:
                print(f"创建日志目录失败 {dir_path}: {str(e)}")
                raise

    start_time = time.time()
    print(f"\n[{datetime.now()}] 推荐系统启动中...")
    sys.stdout.flush()  # 立即刷新输出缓冲区
    
    parser = argparse.ArgumentParser(description='推荐系统服务')
    parser.add_argument('--log-file', type=str, help='日志文件路径')
    parser.add_argument('--debug', action='store_true', help='是否启用调试模式')
    parser.add_argument('--init-db', action='store_true', help='是否初始化数据库表结构(默认不初始化)')
    parser.add_argument('--no-scheduler', action='store_true', help='是否禁用调度任务服务')
    parser.add_argument('--run-tasks', action='store_true', help='立即执行三个主要任务(热门内容更新、用户兴趣模型更新、全量数据分析)')
    args = parser.parse_args()
    
    print(f"[{datetime.now()}] 启动参数: {args}")
    print(f"[{datetime.now()}] 运行环境: Python {sys.version}, OS: {os.name}")
    sys.stdout.flush()  # 立即刷新输出缓冲区
    
    # 重定向日志到文件
    if args.log_file:
        try:
            print(f"[{datetime.now()}] 准备重定向日志到文件: {args.log_file}")
            sys.stdout.flush()
            
            log_dir = os.path.dirname(args.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                print(f"[{datetime.now()}] 创建日志目录: {log_dir}")
                sys.stdout.flush()
                
            # 禁用缓冲区的方式打开文件
            log_file = open(args.log_file, 'a', buffering=1)
            sys.stdout = log_file
            sys.stderr = log_file
            print(f"\n[{datetime.now()}] 系统启动，日志重定向到 {args.log_file}")
            sys.stdout.flush()
        except Exception as e:
            print(f"[{datetime.now()}] 日志重定向失败: {str(e)}")
            import traceback
            print(f"错误详情: {traceback.format_exc()}")
            sys.stdout.flush()
    
    # 注册信号处理函数
    try:
        print(f"[{datetime.now()}] 正在注册信号处理函数...")
        sys.stdout.flush()
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        print(f"[{datetime.now()}] 信号处理函数已注册")
        sys.stdout.flush()
    except Exception as e:
        print(f"[{datetime.now()}] 注册信号处理函数失败: {str(e)}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        sys.stdout.flush()
    
    # 初始化数据库连接池
    try:
        print(f"[{datetime.now()}] 正在初始化数据库连接池...")
        sys.stdout.flush()
        db_pool = get_db_pool()
        print(f"[{datetime.now()}] 数据库连接池初始化完成")
        sys.stdout.flush()
    except Exception as e:
        print(f"[{datetime.now()}] 数据库连接池初始化失败: {str(e)}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        sys.stdout.flush()
    
    # 初始化数据库表结构 (只有指定参数时才执行)
    if args.init_db:
        init_tables()
        sys.stdout.flush()
    else:
        print(f"[{datetime.now()}] 跳过数据库表结构初始化，如需初始化请使用 --init-db 参数")
        sys.stdout.flush()
    
    # 如果指定了立即执行任务的参数，则执行所有主要任务
    if args.run_tasks:
        print(f"[{datetime.now()}] 检测到--run-tasks参数，准备立即执行所有主要任务...")
        sys.stdout.flush()
        run_all_tasks()
        sys.stdout.flush()
    
    # 启动定时任务线程
    scheduler_thread = None  # 初始化为 None
    if not args.no_scheduler:
        try:
            print(f"[{datetime.now()}] 启动调度任务服务...")
            sys.stdout.flush()
            scheduler = TaskScheduler()
            scheduler_thread = threading.Thread(target=scheduler.start, daemon=True)
            scheduler_thread.start()
            print(f"[{datetime.now()}] 调度任务线程已启动")
            sys.stdout.flush()
        except Exception as e:
            print(f"[{datetime.now()}] 启动调度任务服务失败: {str(e)}")
            import traceback
            print(f"错误详情: {traceback.format_exc()}")
            sys.stdout.flush()
    else:
        print(f"[{datetime.now()}] 已禁用调度任务服务")
        sys.stdout.flush()
    
    # 输出启动完成信息
    startup_time = time.time() - start_time
    print(f"[{datetime.now()}] 系统启动完成，耗时: {startup_time:.2f}秒")
    print(f"[{datetime.now()}] 系统运行中，按Ctrl+C终止...")
    sys.stdout.flush()
    
    # 主线程等待退出信号
    try:
        last_status_time = time.time()
        iterations = 0
        while not should_exit:
            iterations += 1
            current_time = time.time()
            
            # 检查是否需要输出状态日志
            if current_time - last_status_time > 60:
                print(f"[{datetime.now()}] 系统运行正常，已运行: {(current_time - start_time) / 60:.1f}分钟")
                sys.stdout.flush()
                last_status_time = current_time
            
            # 每100次循环输出一个调试日志点，用于确认主循环正在运行
            if iterations % 100 == 0:
                print(f"[{datetime.now()}] 主循环健康检查 - 迭代次数: {iterations}")
                sys.stdout.flush()
            
            # 如果有调度器线程，检查它是否还活着
            if scheduler_thread:
                scheduler_thread.join(1.0)
            else:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] 收到键盘中断，正在退出...")
        sys.stdout.flush()
    except Exception as e:
        print(f"[{datetime.now()}] 主循环异常: {str(e)}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        sys.stdout.flush()
    finally:
        total_runtime = time.time() - start_time
        print(f"[{datetime.now()}] 系统已安全退出，总运行时间: {total_runtime:.2f}秒")
        sys.stdout.flush()

if __name__ == '__main__':
    main()
