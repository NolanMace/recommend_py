import os
import sys
import argparse
import logging
import logging.handlers
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
import signal

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入项目组件
from recommender.engine import RecommendationEngine
from exposure.pool_manager import ExposurePoolManager
from hot_topics.generator import HotTopicGenerator
from scheduler.task_scheduler import get_task_scheduler
from cache.cache_manager import get_cache_manager
from database import db_manager  # 假设已有数据库模块

# 全局变量
app_running = True
logger = None


def setup_logging(log_file=None, debug=False):
    """设置日志系统"""
    global logger
    
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志格式
    log_format = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 创建日志处理器
    handlers = []
    
    # 控制台日志
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    handlers.append(console_handler)
    
    # 文件日志
    if log_file:
        # 确保路径存在
        log_path = os.path.join(log_dir, log_file)
        # 每天轮转，保留7天
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_path,
            when='midnight',
            interval=1,
            backupCount=7,
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        handlers.append(file_handler)
    
    # 设置日志级别
    log_level = logging.DEBUG if debug else logging.INFO
    
    # 配置根日志器
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
    
    # 创建应用日志器
    logger = logging.getLogger('recommend_system')
    
    # 降低其他库的日志级别
    logging.getLogger('schedule').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return logger


def init_database(force=False):
    """初始化数据库"""
    logger.info("正在初始化数据库...")
    try:
        db_manager.init_database(force=force)
        logger.info("数据库初始化完成")
        return True
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        return False


def init_components():
    """初始化各组件"""
    logger.info("正在初始化系统组件...")
    
    components = {}
    
    try:
        # 初始化缓存管理器
        cache_manager = get_cache_manager(
            use_redis=True,
            redis_config={
                'host': 'localhost',
                'port': 6379,
                'db': 0
            }
        )
        components['cache_manager'] = cache_manager
        logger.info("缓存管理器初始化完成")
        
        # 初始化推荐引擎
        recommendation_engine = RecommendationEngine(model_dir='./models')
        components['recommendation_engine'] = recommendation_engine
        logger.info("推荐引擎初始化完成")
        
        # 初始化曝光池管理器
        exposure_manager = ExposurePoolManager(db_manager=db_manager)
        components['exposure_manager'] = exposure_manager
        logger.info("曝光池管理器初始化完成")
        
        # 初始化热点生成器
        hot_topic_generator = HotTopicGenerator(
            db_manager=db_manager,
            cache_manager=cache_manager
        )
        components['hot_topic_generator'] = hot_topic_generator
        logger.info("热点生成器初始化完成")
        
        # 初始化任务调度器
        task_scheduler = get_task_scheduler(
            num_workers=4,
            log_file=os.path.join('logs', 'tasks.log')
        )
        components['task_scheduler'] = task_scheduler
        logger.info("任务调度器初始化完成")
        
        return components
    except Exception as e:
        logger.error(f"组件初始化失败: {e}")
        return None


def setup_scheduled_tasks(components):
    """设置定时任务"""
    logger.info("正在设置定时任务...")
    
    scheduler = components['task_scheduler']
    recommendation_engine = components['recommendation_engine']
    hot_topic_generator = components['hot_topic_generator']
    exposure_manager = components['exposure_manager']
    
    # 模型更新任务 - 每天凌晨00:30
    scheduler.schedule_task(
        update_recommendation_model,
        "更新推荐模型",
        'daily',
        '00:30',
        components=components,
        priority=10
    )
    
    # 热点话题生成任务 - 每5分钟
    scheduler.schedule_task(
        generate_hot_topics,
        "生成热点话题",
        'interval',
        300,  # 5分钟
        components=components,
        priority=8
    )
    
    # 批量推荐生成任务 - 每60分钟
    scheduler.schedule_task(
        generate_batch_recommendations,
        "生成批量推荐",
        'interval',
        3600,  # 60分钟
        components=components,
        priority=5
    )
    
    # 曝光池刷新任务 - 每10分钟
    scheduler.schedule_task(
        refresh_exposure_pools,
        "刷新曝光池",
        'interval',
        600,  # 10分钟
        components=components,
        priority=7
    )
    
    # 数据清理任务 - 每天凌晨02:00
    scheduler.schedule_task(
        cleanup_data,
        "清理过期数据",
        'daily',
        '02:00',
        components=components,
        priority=3
    )
    
    logger.info("定时任务设置完成")


def update_recommendation_model(components):
    """更新推荐模型"""
    logger.info("开始更新推荐模型...")
    
    try:
        # 获取必要的组件
        recommendation_engine = components['recommendation_engine']
        
        # 从数据库获取训练数据
        data = db_manager.get_model_training_data()
        
        # 训练模型
        recommendation_engine.train_model(data)
        
        logger.info("推荐模型更新成功")
        return True
    except Exception as e:
        logger.error(f"推荐模型更新失败: {e}")
        return False


def generate_hot_topics(components):
    """生成热点话题"""
    logger.info("开始生成热点话题...")
    
    try:
        # 获取热点生成器
        hot_topic_generator = components['hot_topic_generator']
        
        # 生成热点话题
        hot_topics = hot_topic_generator.generate_hot_topics(force=True)
        
        logger.info(f"热点话题生成成功，共{len(hot_topics)}条")
        return len(hot_topics)
    except Exception as e:
        logger.error(f"热点话题生成失败: {e}")
        return 0


def generate_batch_recommendations(components):
    """生成批量推荐"""
    logger.info("开始生成批量推荐...")
    
    try:
        # 获取必要的组件
        recommendation_engine = components['recommendation_engine']
        cache_manager = components['cache_manager']
        
        # 获取活跃用户列表
        active_users = db_manager.get_active_users(days=7, limit=1000)
        
        if not active_users:
            logger.warning("未找到活跃用户，跳过批量推荐")
            return 0
            
        success_count = 0
        
        # 为每个用户生成推荐
        for user in active_users:
            try:
                user_id = user['user_id']
                
                # 获取用户数据
                user_data = db_manager.get_user_data(user_id)
                
                # 生成推荐
                recommendations = recommendation_engine.get_recommendations(
                    user_data, 
                    top_n=50
                )
                
                if recommendations:
                    # 保存到数据库
                    db_manager.save_user_recommendations(user_id, recommendations)
                    
                    # 缓存结果
                    cache_manager.set_user_recommendations(user_id, recommendations)
                    
                    success_count += 1
            except Exception as e:
                logger.error(f"为用户 {user_id} 生成推荐失败: {e}")
                
        logger.info(f"批量推荐生成成功，处理用户数: {success_count}/{len(active_users)}")
        return success_count
    except Exception as e:
        logger.error(f"批量推荐生成失败: {e}")
        return 0


def refresh_exposure_pools(components):
    """刷新曝光池"""
    logger.info("开始刷新曝光池...")
    
    try:
        # 获取曝光池管理器
        exposure_manager = components['exposure_manager']
        
        # 刷新曝光池
        refresh_results = exposure_manager.refresh_pools()
        
        total_posts = sum(refresh_results.values())
        logger.info(f"曝光池刷新成功，共{total_posts}条帖子")
        return total_posts
    except Exception as e:
        logger.error(f"曝光池刷新失败: {e}")
        return 0


def cleanup_data(components):
    """清理过期数据"""
    logger.info("开始清理过期数据...")
    
    try:
        # 清理过期的推荐结果
        rec_count = db_manager.cleanup_recommendations(days=3)
        logger.info(f"已清理过期推荐结果: {rec_count}条")
        
        # 清理过期的曝光历史
        exp_count = db_manager.cleanup_exposures(hours=72)
        logger.info(f"已清理过期曝光历史: {exp_count}条")
        
        # 清理过期的热点历史
        hot_count = db_manager.cleanup_hot_topics_history(days=30)
        logger.info(f"已清理过期热点历史: {hot_count}条")
        
        # 清理内存缓存
        exposure_manager = components['exposure_manager']
        cache_manager = components['cache_manager']
        
        exp_clean_count = exposure_manager.cleanup_exposure_history()
        cache_clean_count = cache_manager.cleanup()
        
        logger.info(f"已清理内存缓存: 曝光历史{exp_clean_count}条，其他缓存{cache_clean_count}条")
        
        return {
            'recommendations': rec_count,
            'exposures': exp_count,
            'hot_topics': hot_count,
            'memory_exposures': exp_clean_count,
            'memory_cache': cache_clean_count
        }
    except Exception as e:
        logger.error(f"数据清理失败: {e}")
        return None


def run_all_tasks(components):
    """立即运行所有主要任务"""
    logger.info("立即运行所有主要任务...")
    
    scheduler = components['task_scheduler']
    
    # 按顺序执行任务
    tasks = [
        (update_recommendation_model, "更新推荐模型"),
        (refresh_exposure_pools, "刷新曝光池"),
        (generate_hot_topics, "生成热点话题"),
        (generate_batch_recommendations, "生成批量推荐")
    ]
    
    results = {}
    
    for task_func, task_name in tasks:
        logger.info(f"执行任务: {task_name}")
        try:
            result = scheduler.execute_task_now(task_func, task_name, components=components)
            results[task_name] = result
            logger.info(f"任务 {task_name} 执行完成")
        except Exception as e:
            logger.error(f"任务 {task_name} 执行失败: {e}")
            results[task_name] = False
            
    return results


def handle_signal(sig, frame):
    """处理信号"""
    global app_running
    
    if sig == signal.SIGINT:
        logger.info("接收到中断信号，正在停止应用...")
    elif sig == signal.SIGTERM:
        logger.info("接收到终止信号，正在停止应用...")
        
    app_running = False


def main():
    """主函数"""
    global app_running
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="推荐系统服务")
    parser.add_argument("--init-db", action="store_true", help="初始化数据库")
    parser.add_argument("--debug", action="store_true", help="开启调试模式")
    parser.add_argument("--no-scheduler", action="store_true", help="不启动调度器")
    parser.add_argument("--log-file", default="recommend_system.log", help="日志文件名")
    parser.add_argument("--run-tasks", action="store_true", help="立即运行所有主要任务")
    parser.add_argument("--no-log", action="store_true", help="不生成日志文件")
    
    args = parser.parse_args()
    
    # 设置日志
    log_file = None if args.no_log else args.log_file
    log = setup_logging(log_file=log_file, debug=args.debug)
    
    # 打印启动信息
    log.info("="*50)
    log.info("推荐系统服务启动中...")
    log.info(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"运行模式: {'调试模式' if args.debug else '正常模式'}")
    log.info(f"调度器: {'禁用' if args.no_scheduler else '启用'}")
    log.info("="*50)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # 初始化数据库（如果需要）
    if args.init_db:
        if not init_database(force=True):
            log.error("数据库初始化失败，退出程序")
            return 1
    
    # 初始化组件
    components = init_components()
    
    if not components:
        log.error("组件初始化失败，退出程序")
        return 1
    
    # 如果需要立即运行任务
    if args.run_tasks:
        run_all_tasks(components)
        if not args.no_scheduler:
            log.info("任务执行完成，继续运行调度器...")
        else:
            log.info("任务执行完成，程序退出")
            return 0
    
    # 启动调度器（如果需要）
    if not args.no_scheduler:
        # 设置定时任务
        setup_scheduled_tasks(components)
        
        # 启动任务调度器
        task_scheduler = components['task_scheduler']
        task_scheduler.start()
        
        log.info("任务调度器已启动，等待执行任务...")
        
        # 主循环
        try:
            while app_running:
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("接收到键盘中断，程序退出...")
        finally:
            log.info("正在停止任务调度器...")
            task_scheduler.stop()
            log.info("程序已停止")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 