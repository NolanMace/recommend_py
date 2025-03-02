# 导入所需模块
import logging
import pymysql
import time
from typing import Dict, List, Any, Optional, Set, Tuple
import threading
from datetime import datetime, timedelta

from config.config_manager import ConfigManager

# 获取日志记录器
logger = logging.getLogger('db_manager')

# 全局数据库连接池
_db_pool = None
# 线程锁
_lock = threading.Lock()

def get_db_connection():
    """获取数据库连接
    
    Returns:
        数据库连接对象
    """
    global _db_pool
    
    try:
        # 使用锁确保线程安全
        with _lock:
            # 如果连接池不存在或已关闭，则创建新的连接池
            if _db_pool is None or _db_pool.open == 0:
                # 从配置文件中加载数据库连接参数
                config = ConfigManager().get_config()
                
                # 初始化连接池
                _db_pool = pymysql.connect(
                    host=config.get('database.mysql.host', 'localhost'),
                    port=config.get('database.mysql.port', 3306),
                    user=config.get('database.mysql.user', 'root'),
                    password=config.get('database.mysql.password', ''),
                    database=config.get('database.mysql.database', 'recommend_system'),
                    charset=config.get('database.mysql.charset', 'utf8mb4'),
                    cursorclass=pymysql.cursors.DictCursor
                )
                
                logger.info("成功创建数据库连接池")
    
    except Exception as e:
        logger.error(f"创建数据库连接池失败: {e}")
        raise e
    
    return _db_pool

def get_user_data(user_id: str) -> Dict:
    """获取用户数据
    
    包括用户基本信息、历史交互和兴趣标签
    
    Args:
        user_id: 用户ID
        
    Returns:
        用户数据字典
    """
    try:
        conn = get_db_connection()
        result = {
            'user_id': user_id,
            'liked_posts': [],
            'viewed_posts': [],
            'interest_tags': []
        }
        
        # 获取用户点赞的帖子
        with conn.cursor() as cursor:
            sql = """
            SELECT post_id FROM user_interactions 
            WHERE user_id = %s AND interaction_type = 'like'
            ORDER BY interaction_time DESC LIMIT 100
            """
            cursor.execute(sql, (user_id,))
            result['liked_posts'] = [row['post_id'] for row in cursor.fetchall()]
            
        # 获取用户浏览的帖子
        with conn.cursor() as cursor:
            sql = """
            SELECT post_id FROM user_interactions 
            WHERE user_id = %s AND interaction_type = 'view'
            ORDER BY interaction_time DESC LIMIT 200
            """
            cursor.execute(sql, (user_id,))
            result['viewed_posts'] = [row['post_id'] for row in cursor.fetchall()]
            
        # 获取用户兴趣标签
        with conn.cursor() as cursor:
            sql = """
            SELECT tag_name FROM user_tags 
            WHERE user_id = %s
            ORDER BY weight DESC
            """
            cursor.execute(sql, (user_id,))
            result['interest_tags'] = [row['tag_name'] for row in cursor.fetchall()]
            
        return result
    except Exception as e:
        logger.error(f"获取用户数据失败: {e}")
        return {'user_id': user_id, 'liked_posts': [], 'viewed_posts': [], 'interest_tags': []}

def record_user_views(user_id: str, post_ids: List[int]) -> bool:
    """记录用户浏览历史
    
    Args:
        user_id: 用户ID
        post_ids: 帖子ID列表
        
    Returns:
        是否成功
    """
    if not post_ids:
        return True
        
    try:
        conn = get_db_connection()
        
        with conn.cursor() as cursor:
            # 批量插入
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            values = [(user_id, post_id, 'view', now) for post_id in post_ids]
            
            sql = """
            INSERT INTO user_interactions
            (user_id, post_id, interaction_type, interaction_time)
            VALUES (%s, %s, %s, %s)
            """
            
            cursor.executemany(sql, values)
            conn.commit()
            
        return True
    except Exception as e:
        logger.error(f"记录用户浏览历史失败: {e}")
        return False

def get_hot_posts(limit: int = 20, exclude_ids: List[int] = None) -> List[Dict]:
    """获取热门帖子
    
    Args:
        limit: 获取数量
        exclude_ids: 排除的帖子ID列表
        
    Returns:
        帖子列表
    """
    try:
        conn = get_db_connection()
        
        # 构建查询条件
        exclude_clause = ""
        params = [limit]
        
        if exclude_ids and len(exclude_ids) > 0:
            placeholders = ','.join(['%s'] * len(exclude_ids))
            exclude_clause = f"AND post_id NOT IN ({placeholders})"
            params = exclude_ids + params
            
        with conn.cursor() as cursor:
            sql = f"""
            SELECT post_id, title, content, publish_time, heat_score 
            FROM posts 
            WHERE status = 'active' {exclude_clause}
            ORDER BY heat_score DESC
            LIMIT %s
            """
            
            cursor.execute(sql, params)
            return cursor.fetchall()
    except Exception as e:
        logger.error(f"获取热门帖子失败: {e}")
        return []

def get_newest_posts(limit: int = 20, exclude_ids: List[int] = None) -> List[Dict]:
    """获取最新帖子
    
    Args:
        limit: 获取数量
        exclude_ids: 排除的帖子ID列表
        
    Returns:
        帖子列表
    """
    try:
        conn = get_db_connection()
        
        # 构建查询条件
        exclude_clause = ""
        params = [limit]
        
        if exclude_ids and len(exclude_ids) > 0:
            placeholders = ','.join(['%s'] * len(exclude_ids))
            exclude_clause = f"AND post_id NOT IN ({placeholders})"
            params = exclude_ids + params
            
        with conn.cursor() as cursor:
            sql = f"""
            SELECT post_id, title, content, publish_time 
            FROM posts 
            WHERE status = 'active' {exclude_clause}
            ORDER BY publish_time DESC
            LIMIT %s
            """
            
            cursor.execute(sql, params)
            return cursor.fetchall()
    except Exception as e:
        logger.error(f"获取最新帖子失败: {e}")
        return []

def get_quality_posts(limit: int = 20, exclude_ids: List[int] = None) -> List[Dict]:
    """获取高质量帖子
    
    根据点赞率和评论数量综合排序
    
    Args:
        limit: 获取数量
        exclude_ids: 排除的帖子ID列表
        
    Returns:
        帖子列表
    """
    try:
        conn = get_db_connection()
        
        # 构建查询条件
        exclude_clause = ""
        params = [limit]
        
        if exclude_ids and len(exclude_ids) > 0:
            placeholders = ','.join(['%s'] * len(exclude_ids))
            exclude_clause = f"AND p.post_id NOT IN ({placeholders})"
            params = exclude_ids + params
            
        with conn.cursor() as cursor:
            sql = f"""
            SELECT p.post_id, p.title, p.content, p.publish_time,
                   COALESCE(s.like_count, 0) as like_count,
                   COALESCE(s.comment_count, 0) as comment_count,
                   COALESCE(s.view_count, 0) as view_count,
                   CASE 
                       WHEN COALESCE(s.view_count, 0) > 0 
                       THEN COALESCE(s.like_count, 0) / COALESCE(s.view_count, 1) 
                       ELSE 0 
                   END as like_ratio
            FROM posts p
            LEFT JOIN post_stats s ON p.post_id = s.post_id
            WHERE p.status = 'active' {exclude_clause}
            ORDER BY (like_ratio * 10 + comment_count * 0.5) DESC
            LIMIT %s
            """
            
            cursor.execute(sql, params)
            return cursor.fetchall()
    except Exception as e:
        logger.error(f"获取高质量帖子失败: {e}")
        return []

def get_related_tags(tags: List[str], limit: int = 10) -> List[str]:
    """获取相关标签
    
    基于给定标签查找相关的其他标签
    
    Args:
        tags: 标签列表
        limit: 获取的相关标签数量限制
        
    Returns:
        相关标签列表
    """
    if not tags:
        return []
        
    try:
        conn = get_db_connection()
        
        placeholders = ','.join(['%s'] * len(tags))
        params = tags + [limit]
        
        with conn.cursor() as cursor:
            # 查找与给定标签共同出现的其他标签
            sql = f"""
            SELECT related_tag, COUNT(*) as cooccurrence
            FROM tag_relations
            WHERE tag IN ({placeholders})
            AND related_tag NOT IN ({placeholders})
            GROUP BY related_tag
            ORDER BY cooccurrence DESC
            LIMIT %s
            """
            
            cursor.execute(sql, params)
            return [row['related_tag'] for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"获取相关标签失败: {e}")
        return []

def save_user_recommendations(user_id: str, recommendations: List[Dict]) -> bool:
    """保存用户推荐结果到数据库
    
    Args:
        user_id: 用户ID
        recommendations: 推荐结果列表
        
    Returns:
        是否保存成功
    """
    if not recommendations:
        return True
        
    try:
        conn = get_db_connection()
        
        # 先删除旧的推荐
        with conn.cursor() as cursor:
            sql = "DELETE FROM user_recommendations WHERE user_id = %s"
            cursor.execute(sql, (user_id,))
        
        # 批量插入新推荐
        with conn.cursor() as cursor:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            values = []
            
            for i, rec in enumerate(recommendations):
                values.append((
                    user_id,
                    rec['post_id'],
                    rec.get('score', 0),
                    rec.get('algorithm', 'unknown'),
                    i+1,  # 顺序
                    now
                ))
            
            sql = """
            INSERT INTO user_recommendations
            (user_id, post_id, score, algorithm, position, create_time)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            cursor.executemany(sql, values)
            conn.commit()
            
        return True
    except Exception as e:
        logger.error(f"保存用户推荐结果失败: {e}")
        return False

def get_active_users(days: int = 7, limit: int = 1000) -> List[Dict]:
    """获取活跃用户列表
    
    Args:
        days: 活跃天数
        limit: 返回用户数量限制
        
    Returns:
        活跃用户列表
    """
    try:
        conn = get_db_connection()
        
        # 计算N天前的日期
        past_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        with conn.cursor() as cursor:
            sql = """
            SELECT DISTINCT user_id, COUNT(*) as activity_count
            FROM user_interactions
            WHERE interaction_time >= %s
            GROUP BY user_id
            ORDER BY activity_count DESC
            LIMIT %s
            """
            
            cursor.execute(sql, (past_date, limit))
            return cursor.fetchall()
    except Exception as e:
        logger.error(f"获取活跃用户失败: {e}")
        return []

def cleanup_recommendations(days: int = 3) -> int:
    """清理过期的推荐结果
    
    Args:
        days: 保留天数，超过此天数的记录将被删除
        
    Returns:
        清理的记录数
    """
    try:
        conn = get_db_connection()
        
        # 计算过期日期
        expire_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        with conn.cursor() as cursor:
            sql = "DELETE FROM user_recommendations WHERE create_time < %s"
            cursor.execute(sql, (expire_date,))
            count = cursor.rowcount
            conn.commit()
            
        return count
    except Exception as e:
        logger.error(f"清理过期推荐结果失败: {e}")
        return 0

def cleanup_exposures(hours: int = 72) -> int:
    """清理过期的曝光历史
    
    Args:
        hours: 保留小时数，超过此小时数的记录将被删除
        
    Returns:
        清理的记录数
    """
    try:
        conn = get_db_connection()
        
        # 计算过期时间
        expire_time = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
        
        with conn.cursor() as cursor:
            sql = "DELETE FROM exposure_history WHERE exposure_time < %s"
            cursor.execute(sql, (expire_time,))
            count = cursor.rowcount
            conn.commit()
            
        return count
    except Exception as e:
        logger.error(f"清理过期曝光历史失败: {e}")
        return 0

def cleanup_hot_topics_history(days: int = 30) -> int:
    """清理过期的热点话题历史
    
    Args:
        days: 保留天数，超过此天数的记录将被删除
        
    Returns:
        清理的记录数
    """
    try:
        conn = get_db_connection()
        
        # 计算过期日期
        expire_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        with conn.cursor() as cursor:
            sql = "DELETE FROM hot_topics_history WHERE create_time < %s"
            cursor.execute(sql, (expire_date,))
            count = cursor.rowcount
            conn.commit()
            
        return count
    except Exception as e:
        logger.error(f"清理过期热点话题历史失败: {e}")
        return 0

def init_database(force: bool = False):
    """初始化数据库，创建必要的表
    
    Args:
        force: 是否强制重建表（慎用）
    """
    try:
        conn = get_db_connection()
        
        with conn.cursor() as cursor:
            # 检查是否强制重建
            if force:
                # 删除已有表（按照依赖关系倒序删除）
                cursor.execute("DROP TABLE IF EXISTS hot_topics_history")
                cursor.execute("DROP TABLE IF EXISTS exposure_history")
                cursor.execute("DROP TABLE IF EXISTS user_recommendations")
                cursor.execute("DROP TABLE IF EXISTS tag_relations")
                cursor.execute("DROP TABLE IF EXISTS user_tags")
                cursor.execute("DROP TABLE IF EXISTS post_stats")
                cursor.execute("DROP TABLE IF EXISTS user_interactions")
                cursor.execute("DROP TABLE IF EXISTS posts")
                cursor.execute("DROP TABLE IF EXISTS users")
                
            # 创建用户表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR(64) PRIMARY KEY,
                username VARCHAR(64) NOT NULL,
                email VARCHAR(128),
                register_time DATETIME NOT NULL,
                last_active_time DATETIME,
                status ENUM('active', 'inactive', 'banned') DEFAULT 'active'
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            
            # 创建帖子表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                post_id BIGINT PRIMARY KEY,
                title VARCHAR(256) NOT NULL,
                content TEXT,
                author_id VARCHAR(64) NOT NULL,
                publish_time DATETIME NOT NULL,
                update_time DATETIME,
                category VARCHAR(64),
                tags VARCHAR(256),
                heat_score FLOAT DEFAULT 0,
                status ENUM('active', 'deleted', 'hidden') DEFAULT 'active',
                INDEX idx_author (author_id),
                INDEX idx_category (category),
                INDEX idx_publish_time (publish_time),
                INDEX idx_heat_score (heat_score)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            
            # 创建用户交互表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_interactions (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(64) NOT NULL,
                post_id BIGINT NOT NULL,
                interaction_type ENUM('view', 'like', 'comment', 'share', 'collect') NOT NULL,
                interaction_time DATETIME NOT NULL,
                INDEX idx_user_time (user_id, interaction_time),
                INDEX idx_post_type (post_id, interaction_type)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            
            # 创建帖子统计表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS post_stats (
                post_id BIGINT PRIMARY KEY,
                view_count INT DEFAULT 0,
                like_count INT DEFAULT 0,
                comment_count INT DEFAULT 0,
                share_count INT DEFAULT 0,
                collect_count INT DEFAULT 0,
                update_time DATETIME NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            
            # 创建用户标签表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_tags (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(64) NOT NULL,
                tag_name VARCHAR(64) NOT NULL,
                weight FLOAT DEFAULT 1.0,
                update_time DATETIME NOT NULL,
                UNIQUE KEY uk_user_tag (user_id, tag_name),
                INDEX idx_user (user_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            
            # 创建标签关系表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS tag_relations (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                tag VARCHAR(64) NOT NULL,
                related_tag VARCHAR(64) NOT NULL,
                strength FLOAT DEFAULT 0,
                update_time DATETIME NOT NULL,
                UNIQUE KEY uk_tag_pair (tag, related_tag),
                INDEX idx_tag (tag)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            
            # 创建用户推荐表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_recommendations (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(64) NOT NULL,
                post_id BIGINT NOT NULL,
                score FLOAT DEFAULT 0,
                algorithm VARCHAR(32),
                position INT NOT NULL,
                is_clicked TINYINT DEFAULT 0,
                click_time DATETIME,
                create_time DATETIME NOT NULL,
                INDEX idx_user_create (user_id, create_time)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            
            # 创建曝光历史表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS exposure_history (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(64) NOT NULL,
                post_id BIGINT NOT NULL,
                source ENUM('recommendation', 'hot_topic', 'search', 'category', 'other') NOT NULL,
                exposure_time DATETIME NOT NULL,
                INDEX idx_user_time (user_id, exposure_time),
                INDEX idx_post_time (post_id, exposure_time)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            
            # 创建热点话题历史表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS hot_topics_history (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                topic VARCHAR(128) NOT NULL,
                heat_score FLOAT NOT NULL,
                hot_post_ids VARCHAR(512),
                create_time DATETIME NOT NULL,
                INDEX idx_create_time (create_time),
                INDEX idx_heat_score (heat_score)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            
            conn.commit()
            
        logger.info("数据库初始化完成")
    except Exception as e:
        logger.error(f"初始化数据库失败: {e}")
        raise 