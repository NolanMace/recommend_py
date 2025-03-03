import os
import logging
import threading
from typing import Dict, Any, Optional, List
from datetime import timedelta

class ConfigManager:
    """配置管理器
    
    负责加载和管理系统配置，提供统一的配置访问接口
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """初始化配置管理器"""
        if self._initialized:
            return
            
        self.logger = logging.getLogger("config_manager")
        
        # 基础配置
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        
        # 初始化配置字典
        self.config = {
            'mysql': {
                'host': 'localhost',      # 数据库地址
                'port': 3306,            # 数据库端口
                'user': 'root',          # 数据库用户名
                'password': 'root',      # 数据库密码
                'database': 'recommend_system',  # 数据库名
                'charset': 'utf8mb4',    # 字符集
                'autocommit': True,      # 自动提交
                'connect_timeout': 10,   # 连接超时（秒）
                'cursorclass': 'DictCursor'  # 返回字典格式结果
            },
            'pool': {
                'mincached': 2,          # 初始空闲连接数
                'maxcached': 5,          # 最大空闲连接数
                'maxconnections': 20,    # 最大连接数
                'blocking': True         # 连接池满时是否阻塞
            },
            'cache': {
                'max_size': 10000,       # 最大缓存条目数
                'ttl': {
                    'hot_topics': timedelta(minutes=5),      # 热点话题缓存时间
                    'user_recommendations': timedelta(hours=1),  # 用户推荐结果缓存时间
                    'user_history': timedelta(days=7),       # 用户历史缓存时间
                    'system_config': timedelta(days=1)       # 系统配置缓存时间
                }
            },
            'recommender': {
                'default_page_size': 20,     # 默认每页推荐数量
                'max_recommendations': 100,   # 单次最大推荐数量
                'min_score': 0.1,            # 最小推荐分数阈值
                'algorithm_weights': {
                    'content_based': 0.3,     # 基于内容推荐权重
                    'collaborative': 0.4,     # 协同过滤权重
                    'hot_topics': 0.3        # 热点推荐权重
                },
                'behavior_weights': {
                    'view': 1,
                    'click': 2,
                    'like': 3,
                    'collect': 4,
                    'comment': 5
                },
                'time_decay': {
                    'half_life_days': 7,      # 半衰期（天）
                    'max_age_days': 30        # 最大年龄（天）
                }
            },
            'exposure': {
                'global_ratio': 0.3,   # 曝光池在总推荐中的比例
                'pools': {
                    'new': {
                        'ratio': 0.5,
                        'max_age_hours': 24,
                        'min_score': 0.1
                    },
                    'hot': {
                        'ratio': 0.3,
                        'max_age_days': 7,
                        'min_heat_score': 100
                    },
                    'quality': {
                        'ratio': 0.2,
                        'min_quality_score': 0.8,
                        'max_age_days': 30
                    }
                }
            },
            'scheduler': {
                'jobs': {
                    'update_hot_topics': {
                        'interval': timedelta(minutes=5),
                        'max_topics': 50
                    },
                    'precalculate_recommendations': {
                        'interval': timedelta(hours=1),
                        'batch_size': 1000
                    },
                    'clean_cache': {
                        'interval': timedelta(hours=6)
                    },
                    'calculate_statistics': {
                        'interval': timedelta(hours=24)
                    }
                },
                'max_workers': 4,  # 最大工作线程数
                'job_timeout': 300  # 任务超时时间（秒）
            },
            'logging': {
                'version': 1,
                'disable_existing_loggers': False,
                'formatters': {
                    'standard': {
                        'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                    }
                },
                'handlers': {
                    'default': {
                        'level': 'INFO',
                        'formatter': 'standard',
                        'class': 'logging.StreamHandler'
                    },
                    'file': {
                        'level': 'INFO',
                        'formatter': 'standard',
                        'class': 'logging.handlers.RotatingFileHandler',
                        'filename': os.path.join(self.log_dir, 'recommend_system.log'),
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
            },
            'api': {
                'rate_limit': {
                    'default': '100/minute',  # 默认速率限制
                    'strict': '10/minute'     # 严格速率限制
                },
                'timeout': 30,  # API超时时间（秒）
                'max_page_size': 100  # 最大页面大小
            }
        }
        
        # 配置观察者列表
        self._observers = []
        
        # 验证配置
        self.validate_config()
        
        self._initialized = True
    
    def validate_config(self):
        """验证配置项的完整性和有效性"""
        required_configs = [
            'mysql', 'pool', 'cache', 'recommender',
            'exposure', 'scheduler', 'logging', 'api'
        ]
        
        # 检查必要配置项是否存在
        for config_name in required_configs:
            if config_name not in self.config:
                raise ValueError(f"缺少必要的配置项: {config_name}")
        
        # 验证MySQL配置
        mysql_required = ['host', 'port', 'user', 'password', 'database']
        for field in mysql_required:
            if field not in self.config['mysql']:
                raise ValueError(f"MySQL配置缺少必要字段: {field}")
        
        # 验证缓存配置
        if 'max_size' not in self.config['cache']:
            raise ValueError("缓存配置缺少 max_size 字段")
        
        # 验证推荐器配置
        recommender_required = ['default_page_size', 'max_recommendations', 'algorithm_weights']
        for field in recommender_required:
            if field not in self.config['recommender']:
                raise ValueError(f"推荐器配置缺少必要字段: {field}")
        
        # 验证曝光池配置
        if 'pools' not in self.config['exposure']:
            raise ValueError("曝光池配置缺少 pools 字段")
        
        # 验证调度器配置
        if 'jobs' not in self.config['scheduler']:
            raise ValueError("调度器配置缺少 jobs 字段")
            
        self.logger.info("配置验证通过")
    
    def get(self, path: str, default: Any = None) -> Any:
        """获取配置项
        
        支持使用点号分隔的路径访问嵌套配置，如 'mysql.host'
        
        Args:
            path: 配置路径
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        if not path:
            return default
            
        parts = path.split('.')
        value = self.config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
                
        return value
    
    def update_config(self, path: str, new_value: Any):
        """更新指定配置项
        
        Args:
            path: 配置路径，如 'mysql.host'
            new_value: 新的配置值
        """
        if not path:
            return
            
        parts = path.split('.')
        config = self.config
        
        # 遍历路径直到最后一个部分
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]
            
        # 更新最后一个部分的值
        config[parts[-1]] = new_value
        
        # 重新验证配置
        try:
            self.validate_config()
        except ValueError as e:
            # 如果验证失败，回滚更改
            self.logger.error(f"配置更新验证失败: {e}")
            raise
            
        # 通知观察者
        self._notify_observers(path)
        
        self.logger.info(f"配置已更新: {path}")
    
    def register_observer(self, observer):
        """注册配置变更观察者
        
        Args:
            observer: 观察者对象，必须实现 config_updated(path, new_value) 方法
        """
        if observer not in self._observers:
            self._observers.append(observer)
    
    def unregister_observer(self, observer):
        """注销配置变更观察者"""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def _notify_observers(self, path: str):
        """通知所有观察者配置已更新
        
        Args:
            path: 更新的配置路径
        """
        value = self.get(path)
        for observer in self._observers:
            try:
                observer.config_updated(path, value)
            except Exception as e:
                self.logger.error(f"通知观察者失败: {e}")
    
    def get_all(self) -> Dict:
        """获取所有配置
        
        Returns:
            Dict: 完整配置字典的副本
        """
        return self.config.copy()

# 全局配置管理器实例
_config_manager = None

def get_config_manager() -> ConfigManager:
    """获取配置管理器实例
    
    Returns:
        ConfigManager: 配置管理器实例
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager 