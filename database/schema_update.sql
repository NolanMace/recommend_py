-- 帖子表结构更新（添加曝光池和热度字段）
ALTER TABLE posts 
ADD COLUMN exposure_pool INT DEFAULT NULL COMMENT '曝光池级别',
ADD COLUMN heat_score INT DEFAULT 0 COMMENT '热度分数',
ADD COLUMN exposure_count INT DEFAULT 0 COMMENT '曝光次数',
ADD INDEX idx_exposure_pool (exposure_pool),
ADD INDEX idx_heat_score (heat_score),
ADD INDEX idx_created_at (created_at);

-- 帖子曝光记录表（新建）
CREATE TABLE IF NOT EXISTS post_exposures (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL COMMENT '用户ID',
    post_id BIGINT NOT NULL COMMENT '帖子ID',
    exposure_time DATETIME NOT NULL COMMENT '曝光时间',
    click_time DATETIME DEFAULT NULL COMMENT '点击时间',
    INDEX idx_user_post (user_id, post_id),
    INDEX idx_exposure_time (exposure_time),
    INDEX idx_user_exposure (user_id, exposure_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='帖子曝光记录表';

-- 热点话题历史记录表（新建）
CREATE TABLE IF NOT EXISTS hot_topics_history (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    generation_time DATETIME NOT NULL COMMENT '生成时间',
    topics_json TEXT NOT NULL COMMENT '热点话题JSON',
    INDEX idx_generation_time (generation_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='热点话题历史记录表';

-- 用户行为相关表索引优化
-- 用户浏览表索引 - idx_user_post_views
-- 查找所有引用user_views表的外键约束
SELECT CONCAT('ALTER TABLE ', table_name, ' DROP FOREIGN KEY ', constraint_name, ';')
INTO @drop_all_fks
FROM information_schema.referential_constraints
WHERE referenced_table_name = 'user_views' AND constraint_schema = DATABASE()
LIMIT 1;

-- 如果找到了外键约束，先删除它
SET @drop_all_fks = IFNULL(@drop_all_fks, 'SELECT 1;');
PREPARE stmt FROM @drop_all_fks;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 确保没有任何外键约束使用此索引
SET foreign_key_checks = 0;

-- 查询索引是否存在
SELECT COUNT(*) INTO @index_exists FROM information_schema.statistics 
WHERE table_schema = DATABASE() AND table_name = 'user_views' AND index_name = 'idx_user_post_views';

-- 如果索引存在就修改它
SET @alter_stmt = IF(@index_exists > 0, 
  'SET foreign_key_checks = 0; ALTER TABLE user_views DROP INDEX idx_user_post_views, ADD INDEX idx_user_post_views (user_id, post_id, created_at); SET foreign_key_checks = 1;', 
  'ALTER TABLE user_views ADD INDEX idx_user_post_views (user_id, post_id, created_at);');
PREPARE stmt FROM @alter_stmt;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 用户浏览表索引 - idx_post_views
SELECT COUNT(*) INTO @index_exists FROM information_schema.statistics 
WHERE table_schema = DATABASE() AND table_name = 'user_views' AND index_name = 'idx_post_views';

SET @drop_stmt = IF(@index_exists > 0, 'DROP INDEX idx_post_views ON user_views', 'SELECT 1');
PREPARE stmt FROM @drop_stmt;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

ALTER TABLE user_views 
ADD INDEX idx_post_views (post_id, created_at);

-- 用户浏览表索引 - idx_views_time
SELECT COUNT(*) INTO @index_exists FROM information_schema.statistics 
WHERE table_schema = DATABASE() AND table_name = 'user_views' AND index_name = 'idx_views_time';

SET @drop_stmt = IF(@index_exists > 0, 'DROP INDEX idx_views_time ON user_views', 'SELECT 1');
PREPARE stmt FROM @drop_stmt;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

ALTER TABLE user_views 
ADD INDEX idx_views_time (created_at);

-- 帖子点赞表索引 - idx_post_likes_time
SELECT COUNT(*) INTO @index_exists FROM information_schema.statistics 
WHERE table_schema = DATABASE() AND table_name = 'post_likes' AND index_name = 'idx_post_likes_time';

SET @drop_stmt = IF(@index_exists > 0, 'DROP INDEX idx_post_likes_time ON post_likes', 'SELECT 1');
PREPARE stmt FROM @drop_stmt;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

ALTER TABLE post_likes 
ADD INDEX idx_post_likes_time (post_id, created_at);

-- 帖子点赞表索引 - idx_user_likes_time
SELECT COUNT(*) INTO @index_exists FROM information_schema.statistics 
WHERE table_schema = DATABASE() AND table_name = 'post_likes' AND index_name = 'idx_user_likes_time';

SET @drop_stmt = IF(@index_exists > 0, 'DROP INDEX idx_user_likes_time ON post_likes', 'SELECT 1');
PREPARE stmt FROM @drop_stmt;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

ALTER TABLE post_likes 
ADD INDEX idx_user_likes_time (user_id, created_at);

-- 帖子收藏表索引 - idx_post_collects_time
SELECT COUNT(*) INTO @index_exists FROM information_schema.statistics 
WHERE table_schema = DATABASE() AND table_name = 'post_collects' AND index_name = 'idx_post_collects_time';

SET @drop_stmt = IF(@index_exists > 0, 'DROP INDEX idx_post_collects_time ON post_collects', 'SELECT 1');
PREPARE stmt FROM @drop_stmt;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

ALTER TABLE post_collects 
ADD INDEX idx_post_collects_time (post_id, created_at);

-- 帖子收藏表索引 - idx_user_collects_time
SELECT COUNT(*) INTO @index_exists FROM information_schema.statistics 
WHERE table_schema = DATABASE() AND table_name = 'post_collects' AND index_name = 'idx_user_collects_time';

SET @drop_stmt = IF(@index_exists > 0, 'DROP INDEX idx_user_collects_time ON post_collects', 'SELECT 1');
PREPARE stmt FROM @drop_stmt;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

ALTER TABLE post_collects 
ADD INDEX idx_user_collects_time (user_id, created_at);

-- 评论表索引优化 - idx_comments_post_id
SELECT COUNT(*) INTO @index_exists FROM information_schema.statistics 
WHERE table_schema = DATABASE() AND table_name = 'comments' AND index_name = 'idx_comments_post_id';

SET @drop_stmt = IF(@index_exists > 0, 'DROP INDEX idx_comments_post_id ON comments', 'SELECT 1');
PREPARE stmt FROM @drop_stmt;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

ALTER TABLE comments 
ADD INDEX idx_comments_post_id (post_id, created_at);

-- 评论表索引优化 - idx_comments_user_id
SELECT COUNT(*) INTO @index_exists FROM information_schema.statistics 
WHERE table_schema = DATABASE() AND table_name = 'comments' AND index_name = 'idx_comments_user_id';

SET @drop_stmt = IF(@index_exists > 0, 'DROP INDEX idx_comments_user_id ON comments', 'SELECT 1');
PREPARE stmt FROM @drop_stmt;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

ALTER TABLE comments 
ADD INDEX idx_comments_user_id (user_id, created_at);

-- 评论表索引优化 - idx_comments_content
SELECT COUNT(*) INTO @index_exists FROM information_schema.statistics 
WHERE table_schema = DATABASE() AND table_name = 'comments' AND index_name = 'idx_comments_content';

SET @drop_stmt = IF(@index_exists > 0, 'DROP INDEX idx_comments_content ON comments', 'SELECT 1');
PREPARE stmt FROM @drop_stmt;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

ALTER TABLE comments 
ADD INDEX idx_comments_content (content(32));

-- 评论点赞表索引 - idx_comment_likes_comment
SELECT COUNT(*) INTO @index_exists FROM information_schema.statistics 
WHERE table_schema = DATABASE() AND table_name = 'comment_likes' AND index_name = 'idx_comment_likes_comment';

SET @drop_stmt = IF(@index_exists > 0, 'DROP INDEX idx_comment_likes_comment ON comment_likes', 'SELECT 1');
PREPARE stmt FROM @drop_stmt;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

ALTER TABLE comment_likes 
ADD INDEX idx_comment_likes_comment (comment_id, created_at);

-- 评论点赞表索引 - idx_comment_likes_user
SELECT COUNT(*) INTO @index_exists FROM information_schema.statistics 
WHERE table_schema = DATABASE() AND table_name = 'comment_likes' AND index_name = 'idx_comment_likes_user';

SET @drop_stmt = IF(@index_exists > 0, 'DROP INDEX idx_comment_likes_user ON comment_likes', 'SELECT 1');
PREPARE stmt FROM @drop_stmt;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

ALTER TABLE comment_likes 
ADD INDEX idx_comment_likes_user (user_id, created_at);

-- 评论点赞表索引 - idx_comment_likes_count
SELECT COUNT(*) INTO @index_exists FROM information_schema.statistics 
WHERE table_schema = DATABASE() AND table_name = 'comment_likes' AND index_name = 'idx_comment_likes_count';

SET @drop_stmt = IF(@index_exists > 0, 'DROP INDEX idx_comment_likes_count ON comment_likes', 'SELECT 1');
PREPARE stmt FROM @drop_stmt;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

ALTER TABLE comment_likes 
ADD INDEX idx_comment_likes_count (like_count);

-- 搜索记录表索引 - idx_user_search
SELECT COUNT(*) INTO @index_exists FROM information_schema.statistics 
WHERE table_schema = DATABASE() AND table_name = 'user_search_records' AND index_name = 'idx_user_search';

SET @drop_stmt = IF(@index_exists > 0, 'DROP INDEX idx_user_search ON user_search_records', 'SELECT 1');
PREPARE stmt FROM @drop_stmt;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

ALTER TABLE user_search_records 
ADD INDEX idx_user_search (user_id, created_at);

-- 搜索记录表索引 - idx_search_keyword
SELECT COUNT(*) INTO @index_exists FROM information_schema.statistics 
WHERE table_schema = DATABASE() AND table_name = 'user_search_records' AND index_name = 'idx_search_keyword';

SET @drop_stmt = IF(@index_exists > 0, 'DROP INDEX idx_search_keyword ON user_search_records', 'SELECT 1');
PREPARE stmt FROM @drop_stmt;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

ALTER TABLE user_search_records 
ADD INDEX idx_search_keyword (search_word(32), created_at);

-- 搜索记录表索引 - idx_search_time
SELECT COUNT(*) INTO @index_exists FROM information_schema.statistics 
WHERE table_schema = DATABASE() AND table_name = 'user_search_records' AND index_name = 'idx_search_time';

SET @drop_stmt = IF(@index_exists > 0, 'DROP INDEX idx_search_time ON user_search_records', 'SELECT 1');
PREPARE stmt FROM @drop_stmt;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

ALTER TABLE user_search_records 
ADD INDEX idx_search_time (created_at);

-- 用户推荐结果表（新建）
CREATE TABLE IF NOT EXISTS user_recommendations (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL COMMENT '用户ID',
    recommendation_time DATETIME NOT NULL COMMENT '推荐时间',
    post_ids TEXT NOT NULL COMMENT '推荐帖子ID列表(JSON格式)',
    source VARCHAR(50) NOT NULL DEFAULT 'hybrid' COMMENT '推荐来源(hybrid/popularity/exposure)',
    expire_time DATETIME NOT NULL COMMENT '过期时间',
    is_read TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否已读取',
    INDEX idx_user_recommendation (user_id, recommendation_time),
    INDEX idx_expire_time (expire_time),
    INDEX idx_is_read (is_read),
    INDEX idx_user_expire (user_id, expire_time, is_read),
    INDEX idx_source (source, recommendation_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户推荐结果表';

-- 当前热点表（新建，仅保存最新的热点数据）
CREATE TABLE IF NOT EXISTS current_hot_topics (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    update_time DATETIME NOT NULL COMMENT '更新时间',
    post_id BIGINT NOT NULL COMMENT '帖子ID',
    title VARCHAR(255) NOT NULL COMMENT '帖子标题',
    heat_score INT NOT NULL DEFAULT 0 COMMENT '热度分数',
    rank_position INT NOT NULL COMMENT '排名位置',
    UNIQUE KEY idx_post_id (post_id),
    INDEX idx_rank (rank_position),
    INDEX idx_update_time (update_time),
    INDEX idx_heat_rank (heat_score, rank_position)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='当前热点表';

-- 推荐系统日志表（新建）
CREATE TABLE IF NOT EXISTS recommend_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL COMMENT '用户ID',
    recommend_time DATETIME NOT NULL COMMENT '推荐时间',
    post_count INT NOT NULL DEFAULT 0 COMMENT '推荐帖子数量',
    process_time INT NOT NULL DEFAULT 0 COMMENT '处理时间(毫秒)',
    INDEX idx_user_time (user_id, recommend_time),
    INDEX idx_recommend_time (recommend_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='推荐日志表';

-- 性能监控表（新建）
CREATE TABLE IF NOT EXISTS performance_metrics (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    metric_name VARCHAR(50) NOT NULL COMMENT '指标名称',
    metric_value FLOAT NOT NULL COMMENT '指标值',
    collection_time DATETIME NOT NULL COMMENT '收集时间',
    details TEXT NULL COMMENT '详细信息(JSON格式)',
    INDEX idx_metric_name (metric_name, collection_time),
    INDEX idx_collection_time (collection_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='性能监控指标表';

-- 并行批处理作业表（新建）
CREATE TABLE IF NOT EXISTS batch_jobs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    job_type VARCHAR(50) NOT NULL COMMENT '作业类型',
    status VARCHAR(20) NOT NULL DEFAULT 'pending' COMMENT '状态(pending/running/completed/failed)',
    start_time DATETIME NULL COMMENT '开始时间',
    end_time DATETIME NULL COMMENT '结束时间',
    params TEXT NULL COMMENT '作业参数(JSON格式)',
    result TEXT NULL COMMENT '作业结果(JSON格式)',
    worker_count INT NOT NULL DEFAULT 1 COMMENT '工作线程数',
    progress FLOAT NOT NULL DEFAULT 0 COMMENT '进度(0-100)',
    error_message TEXT NULL COMMENT '错误信息',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_job_type (job_type, status),
    INDEX idx_status (status, created_at),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='批处理作业表';

-- 帖子标签优化表（新建）
CREATE TABLE IF NOT EXISTS post_tag_vector (
    post_id BIGINT NOT NULL PRIMARY KEY COMMENT '帖子ID',
    tag_vector BLOB NOT NULL COMMENT '标签向量(二进制格式)',
    update_time DATETIME NOT NULL COMMENT '更新时间',
    INDEX idx_update_time (update_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='帖子标签向量表';

-- 用户兴趣向量表（新建）
CREATE TABLE IF NOT EXISTS user_interest_vector (
    user_id BIGINT NOT NULL PRIMARY KEY COMMENT '用户ID',
    interest_vector BLOB NOT NULL COMMENT '兴趣向量(二进制格式)',
    update_time DATETIME NOT NULL COMMENT '更新时间',
    INDEX idx_update_time (update_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户兴趣向量表';

-- 创建帖子全文检索索引（需要FULLTEXT特性支持）
ALTER TABLE posts
ADD FULLTEXT INDEX ft_title_content (post_title, post_content);

-- 创建推荐系统配置表
CREATE TABLE IF NOT EXISTS recommend_configs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    config_key VARCHAR(50) NOT NULL COMMENT '配置键',
    config_value TEXT NOT NULL COMMENT '配置值(JSON格式)',
    description VARCHAR(255) NULL COMMENT '配置描述',
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    UNIQUE KEY uk_config_key (config_key),
    INDEX idx_update_time (update_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='推荐系统配置表';

-- 缓存管理表（新建）
CREATE TABLE IF NOT EXISTS cache_management (
    cache_key VARCHAR(255) NOT NULL PRIMARY KEY COMMENT '缓存键',
    value_type VARCHAR(20) NOT NULL COMMENT '值类型(user_profile/post_feature/model等)',
    expire_time DATETIME NOT NULL COMMENT '过期时间',
    last_access DATETIME NOT NULL COMMENT '最后访问时间',
    hit_count INT NOT NULL DEFAULT 0 COMMENT '命中次数',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_value_type (value_type, expire_time),
    INDEX idx_expire_time (expire_time),
    INDEX idx_last_access (last_access)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='缓存管理表';

-- 添加查询优化的视图
CREATE OR REPLACE VIEW active_users_view AS
SELECT 
    u.user_id,
    COUNT(DISTINCT DATE(v.created_at)) AS active_days_views,
    COUNT(DISTINCT DATE(l.created_at)) AS active_days_likes,
    COUNT(DISTINCT DATE(c.created_at)) AS active_days_collects,
    COUNT(DISTINCT DATE(cm.created_at)) AS active_days_comments,
    COUNT(DISTINCT DATE(cl.created_at)) AS active_days_comment_likes,
    GREATEST(
        COALESCE(MAX(v.created_at), '1970-01-01'), 
        COALESCE(MAX(l.created_at), '1970-01-01'), 
        COALESCE(MAX(c.created_at), '1970-01-01'),
        COALESCE(MAX(cm.created_at), '1970-01-01'),
        COALESCE(MAX(cl.created_at), '1970-01-01')
    ) AS last_active,
    (
        COUNT(v.post_id) + 
        COUNT(l.post_id) * 3 + 
        COUNT(c.post_id) * 5 +
        COUNT(cm.comment_id) * 4 +
        COUNT(cl.comment_id) * 2
    ) AS activity_score
FROM users u
LEFT JOIN user_views v ON u.user_id = v.user_id AND v.created_at > DATE_SUB(NOW(), INTERVAL 30 DAY)
LEFT JOIN post_likes l ON u.user_id = l.user_id AND l.created_at > DATE_SUB(NOW(), INTERVAL 30 DAY)
LEFT JOIN post_collects c ON u.user_id = c.user_id AND c.created_at > DATE_SUB(NOW(), INTERVAL 30 DAY)
LEFT JOIN comments cm ON u.user_id = cm.user_id AND cm.created_at > DATE_SUB(NOW(), INTERVAL 30 DAY)
LEFT JOIN comment_likes cl ON u.user_id = cl.user_id AND cl.created_at > DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY u.user_id
HAVING 
    COUNT(v.post_id) + COUNT(l.post_id) + COUNT(c.post_id) + COUNT(cm.comment_id) + COUNT(cl.comment_id) > 5;

-- 添加热门帖子查询视图
CREATE OR REPLACE VIEW hot_posts_view AS
SELECT 
    p.post_id,
    p.post_title,
    p.heat_score,
    p.exposure_count,
    p.created_at,
    p.view_count,
    p.like_count,
    p.collect_count,
    p.comment_count,
    COUNT(DISTINCT v.user_id) AS unique_viewers,
    COUNT(DISTINCT l.user_id) AS unique_likers,
    COUNT(DISTINCT c.user_id) AS unique_collectors,
    COUNT(DISTINCT cm.user_id) AS unique_commenters,
    (SELECT COUNT(*) FROM comment_likes cl JOIN comments cm ON cl.comment_id = cm.comment_id 
     WHERE cm.post_id = p.post_id) AS comment_likes_count
FROM posts p
LEFT JOIN user_views v ON p.post_id = v.post_id AND v.created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
LEFT JOIN post_likes l ON p.post_id = l.post_id AND l.created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
LEFT JOIN post_collects c ON p.post_id = c.post_id AND c.created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
LEFT JOIN comments cm ON p.post_id = cm.post_id AND cm.created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
WHERE p.created_at > DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY p.post_id, p.post_title, p.heat_score, p.exposure_count, p.created_at, 
         p.view_count, p.like_count, p.collect_count, p.comment_count
ORDER BY p.heat_score DESC;

-- 修正后的热度分数触发器
DELIMITER //

-- 检查触发器是否存在
SELECT COUNT(*) INTO @trigger_exists FROM information_schema.triggers
WHERE trigger_schema = DATABASE() AND trigger_name = 'before_update_post_heat_score';

-- 如果存在则删除
SET @drop_trigger = IF(@trigger_exists > 0, 'DROP TRIGGER before_update_post_heat_score', 'SELECT 1');
PREPARE stmt FROM @drop_trigger;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 创建新的触发器
CREATE TRIGGER before_update_post_heat_score BEFORE UPDATE ON posts
FOR EACH ROW
BEGIN
    -- 仅当相关计数字段变更时才更新热度分数
    IF NEW.view_count != OLD.view_count OR 
       NEW.like_count != OLD.like_count OR 
       NEW.collect_count != OLD.collect_count OR 
       NEW.comment_count != OLD.comment_count THEN
       
        -- 使用与推荐系统相同的热度计算公式
        SET NEW.heat_score = (
            COALESCE(NEW.view_count, 0) + 
            COALESCE(NEW.like_count, 0) * 3 + 
            COALESCE(NEW.collect_count, 0) * 5 + 
            COALESCE(NEW.comment_count, 0) * 2
        );
        
        -- 当热度分数达到阈值时更新曝光池
        IF NEW.heat_score >= 1000 AND (NEW.exposure_pool < 3 OR NEW.exposure_pool IS NULL) THEN
            SET NEW.exposure_pool = 3;
        ELSEIF NEW.heat_score >= 500 AND (NEW.exposure_pool < 2 OR NEW.exposure_pool IS NULL) THEN
            SET NEW.exposure_pool = 2;
        ELSEIF NEW.heat_score >= 100 AND (NEW.exposure_pool < 1 OR NEW.exposure_pool IS NULL) THEN
            SET NEW.exposure_pool = 1;
        END IF;
    END IF;
END //
DELIMITER ;

-- 批量处理任务状态表（新建）
CREATE TABLE IF NOT EXISTS batch_task_status (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    task_type VARCHAR(50) NOT NULL COMMENT '任务类型(recommendation/hot_topics/cleanup等)',
    start_time DATETIME NOT NULL COMMENT '开始时间',
    end_time DATETIME NULL COMMENT '结束时间',
    total_items INT NOT NULL DEFAULT 0 COMMENT '处理项目总数',
    processed_items INT NOT NULL DEFAULT 0 COMMENT '已处理项目数',
    success_count INT NOT NULL DEFAULT 0 COMMENT '成功处理数',
    error_count INT NOT NULL DEFAULT 0 COMMENT '错误数',
    status VARCHAR(20) NOT NULL DEFAULT 'running' COMMENT '状态(running/completed/failed)',
    error_message TEXT NULL COMMENT '错误信息',
    execution_params TEXT NULL COMMENT '执行参数(JSON格式)',
    worker_count INT NOT NULL DEFAULT 1 COMMENT '工作线程数',
    total_time_ms INT NULL COMMENT '总耗时(毫秒)',
    avg_item_time_ms FLOAT NULL COMMENT '平均项目处理时间(毫秒)',
    INDEX idx_task_type (task_type, start_time),
    INDEX idx_status (status, end_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='批量处理任务状态表';

-- 用户行为统计汇总表（新建-用于加速推荐计算）
CREATE TABLE IF NOT EXISTS user_behavior_stats (
    user_id BIGINT NOT NULL PRIMARY KEY COMMENT '用户ID',
    view_count INT NOT NULL DEFAULT 0 COMMENT '浏览数',
    like_count INT NOT NULL DEFAULT 0 COMMENT '点赞数',
    collect_count INT NOT NULL DEFAULT 0 COMMENT '收藏数',
    comment_count INT NOT NULL DEFAULT 0 COMMENT '评论数',
    comment_like_count INT NOT NULL DEFAULT 0 COMMENT '评论点赞数',
    last_view_time DATETIME NULL COMMENT '最后浏览时间',
    last_like_time DATETIME NULL COMMENT '最后点赞时间',
    last_collect_time DATETIME NULL COMMENT '最后收藏时间',
    last_comment_time DATETIME NULL COMMENT '最后评论时间',
    last_comment_like_time DATETIME NULL COMMENT '最后评论点赞时间',
    last_active_time DATETIME NULL COMMENT '最后活跃时间',
    update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_active_time (last_active_time),
    INDEX idx_update_time (update_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户行为统计汇总表';

-- 优化数据库维护的存储过程
DELIMITER //

-- 检查存储过程是否存在
SELECT COUNT(*) INTO @proc_exists FROM information_schema.routines
WHERE routine_schema = DATABASE() AND routine_name = 'optimize_recommend_tables' AND routine_type = 'PROCEDURE';

-- 如果存在则删除
SET @drop_proc = IF(@proc_exists > 0, 'DROP PROCEDURE optimize_recommend_tables', 'SELECT 1');
PREPARE stmt FROM @drop_proc;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 创建新的存储过程
CREATE PROCEDURE optimize_recommend_tables()
BEGIN
    -- 优化评论相关表
    OPTIMIZE TABLE comments, comment_likes;
    
    -- 优化用户行为表
    OPTIMIZE TABLE user_views, post_likes, post_collects, user_search_records;
    
    -- 优化推荐相关表
    OPTIMIZE TABLE user_recommendations, current_hot_topics, hot_topics_history, 
                 post_exposures, post_tag_vector, user_interest_vector;
    
    -- 更新统计信息
    ANALYZE TABLE posts, users, comments, user_views, post_likes, post_collects;
END //
DELIMITER ;

-- 帖子浏览记录表
CREATE TABLE IF NOT EXISTS post_views (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    user_id BIGINT NOT NULL COMMENT '用户ID',
    post_id BIGINT NOT NULL COMMENT '帖子ID',
    view_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '浏览时间',
    duration INT NOT NULL DEFAULT 0 COMMENT '浏览时长(秒)',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    
    -- 索引
    INDEX idx_user_post (user_id, post_id) COMMENT '用户帖子联合索引',
    INDEX idx_post_time (post_id, view_time) COMMENT '帖子时间索引',
    INDEX idx_user_time (user_id, view_time) COMMENT '用户时间索引',
    INDEX idx_created_at (created_at) COMMENT '创建时间索引'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='帖子浏览记录表';

-- 帖子表
CREATE TABLE IF NOT EXISTS posts (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '帖子ID',
    title VARCHAR(255) NOT NULL COMMENT '帖子标题',
    content TEXT NOT NULL COMMENT '帖子内容',
    user_id BIGINT NOT NULL COMMENT '作者ID',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    view_count INT NOT NULL DEFAULT 0 COMMENT '浏览次数',
    like_count INT NOT NULL DEFAULT 0 COMMENT '点赞次数',
    collect_count INT NOT NULL DEFAULT 0 COMMENT '收藏次数',
    comment_count INT NOT NULL DEFAULT 0 COMMENT '评论次数',
    exposure_pool INT DEFAULT NULL COMMENT '曝光池级别',
    heat_score INT DEFAULT 0 COMMENT '热度分数',
    exposure_count INT DEFAULT 0 COMMENT '曝光次数',
    status TINYINT NOT NULL DEFAULT 1 COMMENT '状态：0-删除，1-正常',
    
    -- 索引
    INDEX idx_user_id (user_id) COMMENT '作者索引',
    INDEX idx_created_at (created_at) COMMENT '创建时间索引',
    INDEX idx_heat_score (heat_score) COMMENT '热度分数索引',
    INDEX idx_exposure_pool (exposure_pool) COMMENT '曝光池索引',
    FULLTEXT INDEX ft_title_content (title, content) COMMENT '全文检索索引'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='帖子表';