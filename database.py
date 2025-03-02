# -*- coding: utf-8 -*-
"""
数据库操作模块
"""
import pymysql
from pymysql.cursors import DictCursor
from dbutils.pooled_db import PooledDB
import time
import logging
from functools import wraps
from datetime import datetime
import threading

from config import MYSQL_CONFIG, POOL_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('database')

# 全局数据库连接池
db_pool = None
pool_lock = threading.Lock()

# 查询统计信息
query_stats = {
    'total_queries': 0,
    'slow_queries': 0,
    'error_queries': 0,
    'total_time': 0,
    'transactions': 0
}
stats_lock = threading.Lock()

def get_db_pool():
    """获取数据库连接池（懒加载）"""
    global db_pool
    if db_pool is None:
        with pool_lock:
            print(f"[{datetime.now()}] 准备获取数据库连接池锁...")
            if db_pool is None:
                try:
                    print(f"[{datetime.now()}] 开始初始化数据库连接池...")
                    print(f"[{datetime.now()}] MySQL配置: {MYSQL_CONFIG}")
                    print(f"[{datetime.now()}] 连接池配置: {POOL_CONFIG}")
                    
                    # 创建连接池
                    print(f"[{datetime.now()}] 尝试连接MySQL服务器: {MYSQL_CONFIG.get('host', 'localhost')}:{MYSQL_CONFIG.get('port', 3306)}")
                    db_pool = DBPool(
                        host=MYSQL_CONFIG.get('host', 'localhost'),
                        port=MYSQL_CONFIG.get('port', 3306),
                        user=MYSQL_CONFIG.get('user', 'root'),
                        password=MYSQL_CONFIG.get('password', ''),
                        database=MYSQL_CONFIG.get('database', ''),
                        charset=MYSQL_CONFIG.get('charset', 'utf8mb4'),
                        cursorclass=DictCursor if MYSQL_CONFIG.get('cursorclass') == 'DictCursor' else None,
                        **POOL_CONFIG
                    )
                    print(f"[{datetime.now()}] 数据库连接池初始化完成，测试连接...")
                    
                    # 测试连接
                    try:
                        db_pool.query("SELECT 1")
                        print(f"[{datetime.now()}] 数据库连接测试成功")
                    except Exception as e:
                        print(f"[{datetime.now()}] 数据库连接测试失败: {str(e)}")
                        raise
                        
                    logger.info("数据库连接池初始化成功")
                except Exception as e:
                    print(f"[{datetime.now()}] 数据库连接池初始化失败: {str(e)}")
                    import traceback
                    print(f"[{datetime.now()}] 错误详情: {traceback.format_exc()}")
                    logger.error(f"数据库连接池初始化失败: {str(e)}")
                    raise
    return db_pool

def log_query(func):
    """装饰器：记录SQL查询的执行情况"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # 提取SQL语句（通常是第一个参数）
        sql = args[1] if len(args) > 1 else kwargs.get('sql', '未知SQL')
        
        # 如果SQL太长，只显示前100个字符
        log_sql = sql[:100] + "..." if len(sql) > 100 else sql
        
        try:
            with stats_lock:
                query_stats['total_queries'] += 1
                
            result = func(*args, **kwargs)
            
            # 计算执行时间
            query_time = time.time() - start_time
            
            with stats_lock:
                query_stats['total_time'] += query_time
                
                # 记录慢查询
                if query_time > 1.0:  # 超过1秒视为慢查询
                    query_stats['slow_queries'] += 1
                    logger.warning(f"慢查询 ({query_time:.3f}s): {log_sql}")
                elif query_time > 0.1:  # 超过100ms记录警告
                    logger.debug(f"查询较慢 ({query_time:.3f}s): {log_sql}")
            
            return result
        except Exception as e:
            with stats_lock:
                query_stats['error_queries'] += 1
                
            logger.error(f"查询出错 ({time.time() - start_time:.3f}s): {log_sql}, 错误: {str(e)}")
            raise
            
    return wrapper

class DBPool:
    """数据库连接池封装"""
    def __init__(self, **kwargs):
        """初始化连接池"""
        self.pool = PooledDB(creator=pymysql, **kwargs)
        self.stats = {
            'connections_created': 0,
            'connections_reused': 0,
            'query_count': 0,
            'transaction_count': 0
        }
        logger.info("数据库连接池初始化完成")
    
    def _get_connection(self):
        """获取数据库连接"""
        try:
            return self.pool.connection()
        except Exception as e:
            logger.error(f"获取数据库连接失败: {str(e)}")
            self.connected = False
            raise
    
    @log_query
    def query(self, sql, params=None, dict_cursor=True):
        """执行查询SQL，返回查询结果"""
        conn = self._get_connection()
        try:
            cursor_class = pymysql.cursors.DictCursor if dict_cursor else pymysql.cursors.Cursor
            cursor = conn.cursor(cursor_class)
            cursor.execute(sql, params)
            result = cursor.fetchall()
            return result
        except Exception as e:
            logger.error(f"查询执行失败: {str(e)}")
            raise
        finally:
            conn.close()
    
    @log_query
    def execute(self, sql, args=None):
        """执行SQL（无返回结果）"""
        conn = self.pool.connection()
        self.stats['connections_reused'] += 1
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, args)
            conn.commit()
        finally:
            conn.close()
    
    @log_query
    def execute_many(self, sql, args_list):
        """批量执行SQL（无返回结果）"""
        if not args_list:
            return
            
        total = len(args_list)
        logger.info(f"开始批量执行SQL，共 {total} 条数据")
        
        conn = self.pool.connection()
        self.stats['connections_reused'] += 1
        try:
            with conn.cursor() as cursor:
                # 分批执行以提高性能
                batch_size = 1000
                batches = (total + batch_size - 1) // batch_size
                
                for i in range(0, total, batch_size):
                    batch = args_list[i:i+batch_size]
                    batch_num = i // batch_size + 1
                    logger.info(f"正在执行第 {batch_num}/{batches} 批, 包含 {len(batch)} 条数据 ({i+1}-{min(i+batch_size, total)}/{total})")
                    
                    start_time = time.time()
                    cursor.executemany(sql, batch)
                    duration = time.time() - start_time
                    
                    logger.info(f"第 {batch_num}/{batches} 批执行完成，耗时: {duration:.2f}秒，处理速度: {len(batch)/duration if duration > 0 else 0:.1f} 条/秒")
                    
            conn.commit()
            logger.info(f"批量执行SQL完成，共 {total} 条数据")
        finally:
            conn.close()
            
    # 添加一个新方法用于执行SQL脚本
    @log_query
    def execute_script(self, script, batch_size=50):
        """执行SQL脚本（多条语句，以分号分隔）"""
        statements = [s.strip() for s in script.split(';') if s.strip()]
        total = len(statements)
        logger.info(f"开始执行SQL脚本，共 {total} 条语句")
        
        start_time = time.time()
        success_count = 0
        error_count = 0
        
        conn = self.pool.connection()
        self.stats['connections_reused'] += 1
        try:
            with conn.cursor() as cursor:
                for i, stmt in enumerate(statements):
                    if i % 10 == 0 or i == total - 1:
                        progress = (i + 1) / total * 100
                        elapsed = time.time() - start_time
                        estimated = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
                        logger.info(f"正在执行SQL语句 [{i+1}/{total}] - {progress:.1f}% 已完成, 预计剩余时间: {estimated:.1f}秒")
                    
                    try:
                        cursor.execute(stmt)
                        success_count += 1
                    except Exception as e:
                        error_count += 1
                        logger.error(f"执行SQL失败 [{i+1}/{total}]: {stmt[:100]}..., 错误: {str(e)}")
                        # 不抛出异常，继续执行下一条
                        
                    # 每处理batch_size条语句就提交一次，避免事务过大
                    if (i + 1) % batch_size == 0:
                        conn.commit()
                        logger.debug(f"已提交 {i+1} 条语句")
                        
            # 最后再提交一次事务
            conn.commit()
            
            # 汇总结果
            duration = time.time() - start_time
            logger.info(f"SQL脚本执行完成: 总计 {total} 条语句, 成功 {success_count}, 失败 {error_count}, 总耗时: {duration:.2f}秒")
            
            return {
                'total': total,
                'success': success_count,
                'error': error_count,
                'duration': duration
            }
        finally:
            conn.close()
    
    def ping(self):
        """检查数据库连接是否可用"""
        try:
            self.query("SELECT 1")
            return True
        except:
            return False
    
    def close(self):
        """关闭连接池"""
        if hasattr(self.pool, 'close'):
            self.pool.close()
        self.connected = False
        logger.info("数据库连接池已关闭")

def get_db_stats():
    """获取数据库统计信息"""
    with stats_lock:
        stats = query_stats.copy()
        
        # 计算平均查询时间
        if stats['total_queries'] > 0:
            stats['avg_query_time'] = stats['total_time'] / stats['total_queries']
        else:
            stats['avg_query_time'] = 0
            
        return stats

def reset_db_stats():
    """重置数据库统计信息"""
    with stats_lock:
        for key in query_stats:
            query_stats[key] = 0
            
    logger.info("数据库统计信息已重置")

def batch_insert(table_name, columns, data_rows, ignore_duplicates=False):
    """批量插入数据
    
    Args:
        table_name: 表名
        columns: 列名列表
        data_rows: 数据行列表
        ignore_duplicates: 是否忽略重复项
    
    Returns:
        影响的行数
    """
    if not data_rows:
        return 0
        
    db = get_db_pool()
    placeholders = ', '.join(['%s'] * len(columns))
    columns_str = ', '.join(columns)
    
    if ignore_duplicates:
        sql = f"INSERT IGNORE INTO {table_name} ({columns_str}) VALUES ({placeholders})"
    else:
        sql = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
        
    return db.execute_many(sql, data_rows)

def bulk_update(table_name, id_column, columns, data_rows):
    """批量更新数据
    
    Args:
        table_name: 表名
        id_column: ID列名
        columns: 需要更新的列名列表
        data_rows: 数据行列表，每行的第一个元素是ID值
    
    Returns:
        影响的行数
    """
    if not data_rows:
        return 0
        
    db = get_db_pool()
    
    # 构建CASE语句
    case_statements = []
    for column in columns:
        case_when = ' '.join([
            f"WHEN {id_column} = {row[0]} THEN %s"
            for row in data_rows
        ])
        case_statements.append(f"{column} = CASE {case_when} ELSE {column} END")
    
    # 构建WHERE子句
    id_list = ', '.join([str(row[0]) for row in data_rows])
    
    # 构建完整SQL
    sql = f"""
    UPDATE {table_name}
    SET {', '.join(case_statements)}
    WHERE {id_column} IN ({id_list})
    """
    
    # 准备参数：每个列的所有值
    params = []
    for i in range(len(columns)):
        col_idx = i + 1  # ID在第0列，数据从第1列开始
        for row in data_rows:
            params.append(row[col_idx])
    
    return db.execute(sql, params)

def dict_to_sql_conditions(conditions, operator='AND'):
    """将字典转换为SQL条件语句
    
    Args:
        conditions: 条件字典，键为列名，值为条件值
        operator: 条件连接符，默认为AND
    
    Returns:
        (条件语句, 参数列表)
    """
    if not conditions:
        return "", []
        
    sql_parts = []
    params = []
    
    for key, value in conditions.items():
        if value is None:
            sql_parts.append(f"{key} IS NULL")
        elif isinstance(value, (list, tuple)):
            placeholders = ', '.join(['%s'] * len(value))
            sql_parts.append(f"{key} IN ({placeholders})")
            params.extend(value)
        elif isinstance(value, dict):
            if 'op' in value and 'value' in value:
                sql_parts.append(f"{key} {value['op']} %s")
                params.append(value['value'])
        else:
            sql_parts.append(f"{key} = %s")
            params.append(value)
    
    return f" {operator} ".join(sql_parts), params

def paginate_query(sql, params=None, page=1, page_size=20):
    """分页查询
    
    Args:
        sql: 基本SQL查询
        params: SQL参数
        page: 页码，从1开始
        page_size: 每页记录数
    
    Returns:
        (记录列表, 总记录数)
    """
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 20
        
    db = get_db_pool()
    
    # 计算总记录数
    count_sql = f"SELECT COUNT(*) as total FROM ({sql}) as temp"
    count_result = db.query(count_sql, params)
    total = count_result[0]['total'] if count_result else 0
    
    # 如果没有记录，直接返回
    if total == 0:
        return [], 0
    
    # 计算分页参数
    offset = (page - 1) * page_size
    
    # 执行分页查询
    paginated_sql = f"{sql} LIMIT %s, %s"
    if params:
        if isinstance(params, (list, tuple)):
            paginated_params = list(params) + [offset, page_size]
        else:
            paginated_params = [params, offset, page_size]
    else:
        paginated_params = [offset, page_size]
    
    records = db.query(paginated_sql, paginated_params)
    
    return records, total

def table_exists(table_name):
    """检查表是否存在
    
    Args:
        table_name: 表名
        
    Returns:
        表是否存在
    """
    db = get_db_pool()
    sql = "SHOW TABLES LIKE %s"
    result = db.query(sql, (table_name,))
    return len(result) > 0

def find_one(table, conditions, columns='*'):
    """查找单条记录
    
    Args:
        table: 表名
        conditions: 条件字典
        columns: 要查询的列，默认为*
    
    Returns:
        单条记录或None
    """
    db = get_db_pool()
    
    # 处理columns参数
    if isinstance(columns, (list, tuple)):
        columns_str = ', '.join(columns)
    else:
        columns_str = columns
    
    # 处理条件
    where_clause, params = dict_to_sql_conditions(conditions)
    if where_clause:
        where_clause = 'WHERE ' + where_clause
    
    # 查询记录
    sql = f"SELECT {columns_str} FROM {table} {where_clause} LIMIT 1"
    result = db.query(sql, params)
    
    return result[0] if result else None