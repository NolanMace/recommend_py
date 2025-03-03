import logging
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from contextlib import contextmanager

import pymysql
from pymysql.cursors import DictCursor
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool

from config.config_manager import get_config_manager

class DatabaseConnectionError(Exception):
    """数据库连接错误"""
    pass

class QueryError(Exception):
    """查询执行错误"""
    pass

class DatabasePool:
    """数据库连接池管理器
    
    提供MySQL连接池管理，支持直接执行SQL和SQLAlchemy ORM操作
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DatabasePool, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化数据库连接池"""
        with self._lock:
            if self._initialized:
                return
            
            self.logger = logging.getLogger("db_pool")
            self.config = get_config_manager()
            
            # 连接池配置
            self.db_config = self.config.get('database', {}).get('mysql', {})
            self.pool_size = self.db_config.get('pool_size', 10)
            self.pool_recycle = self.db_config.get('pool_recycle', 3600)
            self.pool_timeout = self.db_config.get('pool_timeout', 30)
            
            # 连接信息
            self.host = self.db_config.get('host', 'localhost')
            self.port = self.db_config.get('port', 3306)
            self.user = self.db_config.get('user', 'root')
            self.password = self.db_config.get('password', '')
            self.database = self.db_config.get('database', 'recommend_system')
            self.charset = self.db_config.get('charset', 'utf8mb4')
            
            # 连接URL
            self.connection_url = (
                f"mysql+pymysql://{self.user}:{self.password}@"
                f"{self.host}:{self.port}/{self.database}?charset={self.charset}"
            )
            
            # SQLAlchemy引擎和会话
            self.engine = None
            self.session_factory = None
            self.Session = None
            
            # 统计信息
            self.query_count = 0
            self.error_count = 0
            self.last_error_time = None
            self.last_error_message = None
            
            # 熔断器状态
            self.circuit_open = False
            self.failure_count = 0
            self.recovery_time = None
            
            # 熔断器配置
            circuit_config = self.config.get('circuit_breaker', {})
            self.circuit_enabled = circuit_config.get('enabled', True)
            db_circuit = circuit_config.get('services', {}).get('database', {})
            self.failure_threshold = db_circuit.get('failure_threshold', 3)
            self.recovery_timeout = db_circuit.get('recovery_timeout', 60)
            
            # 初始化连接池
            self._initialize_pool()
            
            self._initialized = True
    
    def _initialize_pool(self):
        """初始化SQLAlchemy连接池"""
        try:
            # 创建引擎
            self.engine = create_engine(
                self.connection_url,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                pool_recycle=self.pool_recycle,
                pool_timeout=self.pool_timeout,
                pool_pre_ping=True,  # 连接前ping，确保连接有效
                echo=False
            )
            
            # 创建会话工厂
            self.session_factory = sessionmaker(bind=self.engine)
            
            # 创建线程安全的会话
            self.Session = scoped_session(self.session_factory)
            
            self.logger.info(
                f"数据库连接池初始化成功: {self.host}:{self.port}/{self.database}, "
                f"pool_size={self.pool_size}"
            )
            
            # 重置熔断器
            self.circuit_open = False
            self.failure_count = 0
            self.recovery_time = None
            
            return True
        except Exception as e:
            self.logger.error(f"数据库连接池初始化失败: {e}")
            self._record_error(str(e))
            return False
    
    def _check_circuit_breaker(self):
        """检查熔断器状态
        
        如果熔断器打开，检查是否到达恢复时间
        
        Returns:
            bool: 熔断器是否关闭（可以执行操作）
        """
        if not self.circuit_enabled:
            return True
            
        # 如果熔断器打开
        if self.circuit_open:
            now = time.time()
            
            # 检查是否到达恢复时间
            if self.recovery_time and now >= self.recovery_time:
                self.logger.info("熔断器恢复时间已到，尝试重置连接池")
                if self._initialize_pool():
                    self.logger.info("熔断器已关闭，连接池已重置")
                    return True
                else:
                    # 重置失败，延长恢复时间
                    self.recovery_time = now + self.recovery_timeout
                    self.logger.warning(f"连接池重置失败，熔断器保持打开状态，下次恢复时间: {self.recovery_time}")
                    return False
            else:
                return False
        
        return True
    
    def _record_error(self, error_message: str):
        """记录错误并更新熔断器状态
        
        Args:
            error_message: 错误信息
        """
        self.error_count += 1
        self.last_error_time = time.time()
        self.last_error_message = error_message
        
        if self.circuit_enabled:
            self.failure_count += 1
            
            # 检查是否达到熔断阈值
            if self.failure_count >= self.failure_threshold:
                self.circuit_open = True
                self.recovery_time = time.time() + self.recovery_timeout
                self.logger.warning(
                    f"数据库熔断器已打开，连续失败次数: {self.failure_count}, "
                    f"恢复时间: {self.recovery_timeout}秒后"
                )
    
    def _record_success(self):
        """记录成功操作，重置失败计数"""
        if self.circuit_enabled and self.failure_count > 0:
            self.failure_count = 0
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接
        
        使用上下文管理器自动关闭连接
        
        Yields:
            Connection: 数据库连接对象
            
        Raises:
            DatabaseConnectionError: 连接获取失败
        """
        # 检查熔断器
        if not self._check_circuit_breaker():
            raise DatabaseConnectionError("数据库熔断器已打开，拒绝连接")
            
        conn = None
        try:
            conn = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset=self.charset,
                cursorclass=DictCursor,
                connect_timeout=self.pool_timeout
            )
            self._record_success()
            yield conn
        except Exception as e:
            self._record_error(str(e))
            raise DatabaseConnectionError(f"获取数据库连接失败: {e}")
        finally:
            if conn:
                conn.close()
    
    @contextmanager
    def get_session(self):
        """获取SQLAlchemy会话
        
        使用上下文管理器自动提交或回滚事务
        
        Yields:
            Session: SQLAlchemy会话对象
            
        Raises:
            DatabaseConnectionError: 会话获取失败
        """
        # 检查熔断器
        if not self._check_circuit_breaker():
            raise DatabaseConnectionError("数据库熔断器已打开，拒绝连接")
            
        session = None
        try:
            session = self.Session()
            yield session
            session.commit()
            self._record_success()
        except Exception as e:
            if session:
                session.rollback()
            self._record_error(str(e))
            raise
        finally:
            if session:
                session.close()
    
    def execute_query(self, query: str, params: Dict = None) -> List[Dict]:
        """执行查询SQL
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            List[Dict]: 查询结果列表
            
        Raises:
            QueryError: 查询执行失败
        """
        self.query_count += 1
        
        with self.get_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(query, params or {})
                    return cursor.fetchall()
            except Exception as e:
                raise QueryError(f"查询执行失败: {e}")
    
    def execute_update(self, query: str, params: Dict = None) -> int:
        """执行更新SQL
        
        Args:
            query: SQL更新语句
            params: 更新参数
            
        Returns:
            int: 影响的行数
            
        Raises:
            QueryError: 更新执行失败
        """
        self.query_count += 1
        
        with self.get_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    affected_rows = cursor.execute(query, params or {})
                    conn.commit()
                    return affected_rows
            except Exception as e:
                conn.rollback()
                raise QueryError(f"更新执行失败: {e}")
    
    def execute_batch(self, query: str, params_list: List[Dict]) -> int:
        """批量执行SQL
        
        Args:
            query: SQL语句
            params_list: 参数列表
            
        Returns:
            int: 总影响行数
            
        Raises:
            QueryError: 批量执行失败
        """
        self.query_count += 1
        
        with self.get_connection() as conn:
            try:
                total_affected = 0
                with conn.cursor() as cursor:
                    for params in params_list:
                        affected_rows = cursor.execute(query, params)
                        total_affected += affected_rows
                    conn.commit()
                    return total_affected
            except Exception as e:
                conn.rollback()
                raise QueryError(f"批量执行失败: {e}")
    
    def execute_many(self, query: str, params_list: List[Dict]) -> int:
        """使用executemany批量执行SQL
        
        Args:
            query: SQL语句
            params_list: 参数列表
            
        Returns:
            int: 总影响行数
            
        Raises:
            QueryError: 批量执行失败
        """
        self.query_count += 1
        
        with self.get_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    affected_rows = cursor.executemany(query, params_list)
                    conn.commit()
                    return affected_rows
            except Exception as e:
                conn.rollback()
                raise QueryError(f"批量执行失败: {e}")
    
    def get_stats(self) -> Dict:
        """获取数据库连接池统计信息
        
        Returns:
            Dict: 统计信息字典
        """
        stats = {
            'query_count': self.query_count,
            'error_count': self.error_count,
            'circuit_open': self.circuit_open,
            'failure_count': self.failure_count
        }
        
        if self.last_error_time:
            stats['last_error_time'] = self.last_error_time
            stats['last_error_message'] = self.last_error_message
            
        if self.recovery_time:
            stats['recovery_time'] = self.recovery_time
            
        # 添加SQLAlchemy连接池信息
        if self.engine:
            pool_status = self.engine.pool.status()
            stats.update({
                'pool_size': self.pool_size,
                'checkedin': pool_status.checkedin,
                'checkedout': pool_status.checkedout,
                'checkedout_overflow': pool_status.checkedout_overflow,
                'overflow': pool_status.overflow
            })
            
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.query_count = 0
        self.error_count = 0
        self.last_error_time = None
        self.last_error_message = None

# 全局数据库连接池实例
_db_pool = None

def get_db_pool() -> DatabasePool:
    """获取数据库连接池实例
    
    Returns:
        DatabasePool: 数据库连接池实例
    """
    global _db_pool
    if _db_pool is None:
        _db_pool = DatabasePool()
    return _db_pool 