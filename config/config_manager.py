import os
import logging
import threading
from typing import Dict, Any, Optional, List
from datetime import timedelta
from .config import (
    RECOMMENDER_CONFIG,
    EXPOSURE_CONFIG,
    SCHEDULER_CONFIG,
    LOGGING_CONFIG,
    API_CONFIG
)

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
        self._config = {
            'recommender': RECOMMENDER_CONFIG,
            'exposure': EXPOSURE_CONFIG,
            'scheduler': SCHEDULER_CONFIG,
            'logging': LOGGING_CONFIG,
            'api': API_CONFIG
        }
        
        # 配置观察者列表
        self._observers = []
        
        # 验证配置
        self.validate_config()
        
        self._initialized = True
    
    def validate_config(self):
        """验证配置项的完整性和有效性"""
        required_configs = [
            'recommender', 'exposure', 'scheduler', 'logging', 'api'
        ]
        
        # 检查必要配置项是否存在
        for config_name in required_configs:
            if config_name not in self._config:
                raise ValueError(f"缺少必要的配置项: {config_name}")
        
        # 验证推荐器配置
        recommender_required = ['default_page_size', 'max_recommendations', 'algorithm_weights']
        for field in recommender_required:
            if field not in self._config['recommender']:
                raise ValueError(f"推荐器配置缺少必要字段: {field}")
        
        # 验证曝光池配置
        if 'pools' not in self._config['exposure']:
            raise ValueError("曝光池配置缺少 pools 字段")
        
        # 验证调度器配置
        if 'jobs' not in self._config['scheduler']:
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
        value = self._config
        
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
        config = self._config
        
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
        return self._config.copy()

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