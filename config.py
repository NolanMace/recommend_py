# -*- coding: utf-8 -*-
"""
推荐系统全局配置
"""

MYSQL_CONFIG = {
    'host': 'rm-bp1w3lxy9t0790m4jvo.mysql.rds.aliyuncs.com',      # 数据库地址
    'port': 3306,             # 数据库端口（必须明确指定）
    'user': 'hwn205', # 专用数据库账号（非root）
    'password': 'fuqqym-qemnYc-4heptu',  # 密码需包含特殊字符
    'database': 'pink',   # 数据库名
    'charset': 'utf8mb4',     # 支持中文及表情符号
    'autocommit': True,       # 自动提交事务
    'connect_timeout': 10,    # 连接超时时间（秒）
    'cursorclass': 'DictCursor'  # 返回字典格式结果
}

# 连接池配置（单独定义）
POOL_CONFIG = {
    'mincached': 2,    # 初始空闲连接
    'maxcached': 5,    # 最大空闲连接
    'maxconnections': 20,  # 最大活跃连接
    'blocking': True   # 连接池满时等待
}

# 行为权重配置
BEHAVIOR_WEIGHTS = {
    'view': 1,
    'click': 2,
    'like': 3,
    'collect': 4,
    'comment': 5
}

# TF-IDF特征配置
TFIDF_PARAMS = {
    'max_features': 1000,
    'stop_words': 'english'
}

# 推荐策略配置
RECOMMEND_CONFIG = {
    'top_n': 100,          # 推荐数量
    'hot_days': 7,        # 热门帖子计算天数
    'recent_days': 30     # 用户行为有效期
}

# 曝光池配置
EXPOSURE_CONFIG = {
    'global_ratio': 0.3,   # 曝光池推荐在总推荐中的比例
    # 曝光池定义（按热度分层）
    'pools': {
        # 第一曝光池：低热度新帖
        1: {
            'ratio': 0.5,            # 在曝光池推荐中的比例
            'heat_threshold': 1,     # 最低热度要求
            'max_exposures': 100,    # 最大曝光次数
            'max_age_days': 7,       # 最大帖子年龄（天）
        },
        # 第二曝光池：中热度帖子
        2: {
            'ratio': 0.3,
            'heat_threshold': 100,
            'max_exposures': 500,
            'max_age_days': 14,
        },
        # 第三曝光池：高热度帖子
        3: {
            'ratio': 0.2,
            'heat_threshold': 500,
            'max_exposures': 1000,
            'max_age_days': 30,
        }
    }
}

# 热点生成配置
HOT_TOPICS_CONFIG = {
    'interval_minutes': 5,     # 热点生成间隔（分钟）
    'count': 50,               # 每次生成热点数量
    'min_heat_score': 50       # 最低热度阈值
}

# 数据库存储配置
DATABASE_CONFIG = {
    'recommendation_expire_hours': 24,  # 推荐结果过期时间（小时）
    'active_user_days': 7,             # 活跃用户时间范围（天）
    'batch_size': 100,                 # 批量处理用户数
    'cleanup_days': 30,                # 清理过期数据的天数
    'max_recommendations_per_user': 3,  # 每个用户最多保留的推荐结果数
    'max_workers': 4,                  # 并行处理的工作线程数
    'db_pool_size': 20                 # 数据库连接池大小
}

# 模型训练配置
MODEL_TRAINING_CONFIG = {
    'max_iterations': 100,        # 模型最大迭代次数
    'convergence_threshold': 0.001,  # 收敛阈值
    'learning_rate': 0.01,        # 学习率
    'regularization': 0.1,        # 正则化参数
    'train_test_split': 0.2,      # 训练/测试数据比例
    'random_seed': 42             # 随机种子，确保结果可复现
}

