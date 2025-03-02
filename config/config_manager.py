import os
import logging
import threading
import yaml
from typing import Dict, Any, Optional, List

class ConfigManager:
    """配置管理器
    
    负责加载和管理系统配置，支持从YAML文件加载配置
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
    
    def __init__(self, config_path: str = None):
        """初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认为None，将使用默认路径
        """
        if self._initialized:
            return
            
        self.logger = logging.getLogger("config_manager")
        
        # 默认配置文件路径
        self.config_dir = os.path.dirname(os.path.abspath(__file__))
        self.default_config_path = os.path.join(self.config_dir, "default_config.yaml")
        
        # 用户配置文件路径
        self.user_config_path = config_path or os.environ.get(
            "RECOMMEND_CONFIG_PATH", 
            os.path.join(self.config_dir, "config.yaml")
        )
        
        # 配置数据
        self.config_data = {}
        
        # 加载配置
        self._load_config()
        
        self._initialized = True
    
    def _load_config(self):
        """加载配置文件
        
        首先加载默认配置，然后加载用户配置（如果存在）
        """
        # 加载默认配置
        try:
            with open(self.default_config_path, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f) or {}
                self.logger.info(f"已加载默认配置: {self.default_config_path}")
        except Exception as e:
            self.logger.error(f"加载默认配置失败: {e}")
            self.config_data = {}
        
        # 加载用户配置（如果存在）
        if os.path.exists(self.user_config_path):
            try:
                with open(self.user_config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) or {}
                    # 递归合并配置
                    self._merge_config(self.config_data, user_config)
                    self.logger.info(f"已加载用户配置: {self.user_config_path}")
            except Exception as e:
                self.logger.error(f"加载用户配置失败: {e}")
    
    def _merge_config(self, base: Dict, override: Dict):
        """递归合并配置
        
        Args:
            base: 基础配置
            override: 覆盖配置
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def reload(self):
        """重新加载配置"""
        self._load_config()
        self.logger.info("配置已重新加载")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项
        
        Args:
            key: 配置键，支持点号分隔的路径，如 'database.mysql.host'
            default: 默认值，当配置项不存在时返回
            
        Returns:
            Any: 配置值或默认值
        """
        if not key:
            return default
            
        # 处理点号分隔的路径
        parts = key.split('.')
        value = self.config_data
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
                
        return value
    
    def get_section(self, section: str) -> Dict:
        """获取配置节
        
        Args:
            section: 配置节名称
            
        Returns:
            Dict: 配置节或空字典
        """
        return self.get(section, {})
    
    def set(self, key: str, value: Any):
        """设置配置项
        
        Args:
            key: 配置键，支持点号分隔的路径
            value: 配置值
        """
        if not key:
            return
            
        # 处理点号分隔的路径
        parts = key.split('.')
        config = self.config_data
        
        # 遍历路径，创建必要的嵌套字典
        for i, part in enumerate(parts[:-1]):
            if part not in config or not isinstance(config[part], dict):
                config[part] = {}
            config = config[part]
            
        # 设置最终值
        config[parts[-1]] = value
    
    def save_user_config(self):
        """保存用户配置到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.user_config_path), exist_ok=True)
            
            with open(self.user_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, allow_unicode=True)
                
            self.logger.info(f"用户配置已保存: {self.user_config_path}")
            return True
        except Exception as e:
            self.logger.error(f"保存用户配置失败: {e}")
            return False
    
    def get_all(self) -> Dict:
        """获取所有配置
        
        Returns:
            Dict: 完整配置字典
        """
        return self.config_data.copy()

# 全局配置管理器实例
_config_manager = None

def get_config_manager(config_path: str = None) -> ConfigManager:
    """获取配置管理器实例
    
    Args:
        config_path: 配置文件路径，默认为None
        
    Returns:
        ConfigManager: 配置管理器实例
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager 