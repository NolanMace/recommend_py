# -*- coding: utf-8 -*-
"""
推荐系统配置文件
所有配置项都在此文件中定义，便于统一管理和修改
"""
import os
from datetime import timedelta

# 基础配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# 热点话题配置
HOT_TOPICS_CONFIG = {
    'max_topics': 50,  # 最大热点话题数量
    'min_score': 100,  # 最小热度分数
    'time_window_days': 7,  # 时间窗口（天）
    'update_interval': 300,  # 更新间隔（秒）
    'cache_ttl': 300,  # 缓存时间（秒）
    'weights': {
        'view': 1,      # 浏览权重
        'like': 3,      # 点赞权重
        'collect': 5,   # 收藏权重
        'comment': 2    # 评论权重
    }
}

# 数据库配置
MYSQL_CONFIG = {
    'host': 'rm-bp1w3lxy9t0790m4j.mysql.rds.aliyuncs.com',      # 数据库地址
    'port': 3306,            # 数据库端口
    'user': 'hwn205',          # 数据库用户名
    'password': 'fuqqym-qemnYc-4heptu',      # 数据库密码
    'database': 'pink',  # 数据库名
    'charset': 'utf8mb4',    # 字符集
    'autocommit': True,      # 自动提交
    'connect_timeout': 30,   # 连接超时（秒）增加到30秒
    'cursorclass': 'DictCursor'  # 返回字典格式结果
}

# 数据库连接池配置
POOL_CONFIG = {
    'mincached': 1,          # 初始空闲连接数减少到1
    'maxcached': 3,          # 最大空闲连接数减少到3
    'maxconnections': 10,    # 最大连接数减少到10
    'blocking': True         # 连接池满时是否阻塞
}

# 缓存配置
CACHE_CONFIG = {
    'max_size': 10000,       # 最大缓存条目数
    'ttl': {
        'hot_topics': timedelta(minutes=5),      # 热点话题缓存时间
        'user_recommendations': timedelta(hours=1),  # 用户推荐结果缓存时间
        'user_history': timedelta(days=7),       # 用户历史缓存时间
        'system_config': timedelta(days=1)       # 系统配置缓存时间
    }
}

# 推荐引擎配置
RECOMMENDER_CONFIG = {
    # 基础参数
    'default_page_size': 20,     # 默认每页推荐数量
    'max_recommendations': 100,   # 单次最大推荐数量
    'min_score': 0.1,            # 最小推荐分数阈值
    
    # 算法权重
    'algorithm_weights': {
        'content_based': 0.3,     # 基于内容推荐权重
        'collaborative': 0.4,     # 协同过滤权重
        'hot_topics': 0.3        # 热点推荐权重
    },
    
    # 用户行为权重
    'behavior_weights': {
        'view': 1,
        'click': 2,
        'like': 3,
        'collect': 4,
        'comment': 5
    },
    
    # 时间衰减参数
    'time_decay': {
        'half_life_days': 7,      # 半衰期（天）
        'max_age_days': 30        # 最大年龄（天）
    }
}

# 曝光池配置
EXPOSURE_CONFIG = {
    'global_ratio': 0.3,   # 曝光池在总推荐中的比例
    'pools': {
        # 新内容池
        'new': {
            'ratio': 0.5,
            'max_age_hours': 24,
            'min_score': 0.1
        },
        # 热门内容池
        'hot': {
            'ratio': 0.3,
            'max_age_days': 7,
            'min_heat_score': 100
        },
        # 优质内容池
        'quality': {
            'ratio': 0.2,
            'min_quality_score': 0.8,
            'max_age_days': 30
        }
    }
}

# 任务调度配置
SCHEDULER_CONFIG = {
    'jobs': {
        # 热点话题更新
        'update_hot_topics': {
            'interval': timedelta(minutes=5),
            'max_topics': 50
        },
        # 推荐结果预计算
        'precalculate_recommendations': {
            'interval': timedelta(hours=1),
            'batch_size': 1000
        },
        # 缓存清理
        'clean_cache': {
            'interval': timedelta(hours=6)
        },
        # 数据统计
        'calculate_statistics': {
            'interval': timedelta(hours=24)
        }
    },
    'max_workers': 4,  # 最大工作线程数
    'job_timeout': 300  # 任务超时时间（秒）
}

# 日志配置
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/recommend_system.log',  # 修改为相对路径
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

# API配置
API_CONFIG = {
    'rate_limit': {
        'default': '100/minute',  # 默认速率限制
        'strict': '10/minute'     # 严格速率限制
    },
    'timeout': 30,  # API超时时间（秒）
    'max_page_size': 100  # 最大页面大小
}

# 监控配置
MONITOR_CONFIG = {
    'enabled': True,
    'metrics': {
        'cache_hit_rate': True,
        'api_response_time': True,
        'recommendation_quality': True,
        'system_resources': True
    },
    'alert_thresholds': {
        'error_rate': 0.01,        # 错误率阈值
        'response_time': 1000,     # 响应时间阈值（毫秒）
        'cpu_usage': 80,           # CPU使用率阈值（%）
        'memory_usage': 80         # 内存使用率阈值（%）
    }
}

