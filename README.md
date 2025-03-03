# 帖子推荐系统

## 项目概述

这是一个纯基于 Python 和 MySQL 的高效推荐系统，无需依赖 Redis 或其他第三方服务。系统采用纯 Python 实现各种功能组件，包括内存缓存(LRU)、数据库连接池和任务调度等。通过优化的内存管理和数据库连接机制，提供高效的推荐服务，满足中小规模应用场景需求。

特别适合要求部署简单、依赖少的项目场景。

## 系统架构

```
┌─────────────────────────────────┐
│           用户请求层            │
│     Web API / 应用程序接口      │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│         内存缓存层(LRU)         │
│    热点话题/用户推荐/配置缓存    │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│         混合推荐引擎层          │
│  TF-IDF + 协同过滤 + 混合策略   │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│         曝光池管理层            │
│  新内容/热门/优质/多样性管理    │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│       数据库连接池层            │
│   高效连接管理 + 熔断保护       │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│           MySQL 数据库          │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│         任务调度系统            │
│    定时任务 + 优先级管理        │
└─────────────────────────────────┘
```

## 技术栈

- **编程语言**: Python 3.8+
- **数据库**: MySQL
- **缓存**: 基于 LRU 算法的内存缓存
- **任务调度**: APScheduler
- **Web 框架**: Flask (用于 API 服务)

## 主要特点

- **无 Redis 依赖**: 使用优化的内存 LRU 缓存替代 Redis，简化部署
- **高性能**: 内存缓存和连接池优化性能，减少数据库压力
- **高可靠**: 内置熔断器和异常处理机制，防止系统雪崩
- **可扩展**: 模块化设计，易于扩展和定制
- **易部署**: 简化的部署要求，适合各种环境
- **无限滚动**: 支持用户持续浏览，提供源源不断的内容

## 主要功能模块

### 1. 推荐引擎

- **混合算法**: 结合 TF-IDF 和协同过滤的混合推荐引擎
- **动态策略**: 根据用户浏览深度动态调整推荐策略
- **相似度计算**: 优化的内容相似度和用户兴趣计算

### 2. 曝光池管理

- **多级曝光池**: 新内容池、热门内容池、优质内容池、多样性内容池
- **智能轮转**: 动态调整不同池权重比例
- **曝光控制**: 防止内容重复曝光，确保推荐新鲜度

### 3. 无限滚动推荐策略

- **分层推荐**:
  - 前 3 页：严格个性化推荐
  - 4-10 页：混合推荐（个性化+热门内容）
  - 10 页以后：扩展推荐（降低相似度阈值，扩大兴趣范围）
- **动态扩展**: 随着滚动深度增加，逐步降低内容相似度要求
- **兜底策略**: 智能补充热门、最新、高质量内容

### 4. 内存缓存管理

- **LRU 策略**: 优先淘汰最久未使用的缓存项
- **TTL 控制**: 精细化过期时间控制
- **线程安全**: 多线程环境下的安全访问
- **内存优化**: 自动清理过期数据，控制内存使用

### 5. 数据库连接池

- **连接复用**: 减少频繁创建连接的开销
- **自动重连**: 连接失效时自动重建
- **熔断保护**: 数据库异常时保护系统
- **参数优化**: 连接池大小和超时参数可配置

### 6. 配置管理

- **YAML 配置**: 集中式配置文件管理
- **多环境支持**: 开发、测试、生产环境配置分离
- **动态加载**: 支持配置热更新
- **默认值处理**: 智能处理缺失配置项

### 7. 任务调度

- **定时任务**: 自动执行周期性任务
- **优先级队列**: 重要任务优先执行
- **失败重试**: 自动重试失败任务
- **执行日志**: 详细记录任务执行情况

## 内存缓存设计

系统使用内存 LRU (最近最少使用) 缓存替代传统的 Redis 缓存，具有以下特点：

- **LRU 替换策略**: 当缓存达到最大容量时，优先清除最久未使用的项
- **TTL 支持**: 每个缓存项可设置过期时间
- **自动清理**: 后台线程定期清理过期数据
- **线程安全**: 使用锁机制确保多线程环境下的数据一致性
- **统计数据**: 提供命中率、容量使用等统计信息

针对不同类型的数据，系统采用不同的缓存策略：

- 热点话题: 5 分钟过期时间
- 用户推荐结果: 1 小时过期时间
- 用户浏览历史: 7 天过期时间
- 系统配置: 24 小时过期时间

## 安装与配置

### 安装步骤

1. 克隆代码库

   ```bash
   git clone https://github.com/yourusername/recommend_py.git
   cd recommend_py
   ```

2. 安装依赖

   ```bash
   pip install -r requirements.txt
   ```

3. 创建 MySQL 数据库

   ```bash
   mysql -u root -p
   CREATE DATABASE recommend_system CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   mysql -u root -p recommend_system < schema_update.sql
   ```

4. 配置数据库连接
   创建或编辑 `config/config.yaml` 文件（系统会先使用默认配置）

   ```yaml
   database:
     mysql:
       host: localhost
       port: 3306
       user: your_username
       password: your_password
       database: recommend_system
   ```

5. 初始化数据库表结构

   ```bash
   python app.py --init-db
   ```

6. 运行应用

   ```bash
   # 运行后台服务
   python app.py

   # 或运行API服务
   ./run_api.sh
   ```

### YAML 配置示例

**调整缓存大小**:

```yaml
cache:
  memory:
    max_size: 10000 # 缓存项数量上限
    cleanup_interval: 300 # 清理间隔（秒）
```

**修改连接池设置**:

```yaml
database:
  mysql:
    pool_size: 10
    max_overflow: 20
    timeout: 30
```

## 测试工具

系统提供了测试工具，用于验证推荐功能：

```bash
# 默认测试（使用随机用户）
python test_recommender.py

# 指定用户测试
python test_recommender.py --user 12345

# 批量测试多个用户
python test_recommender.py --batch 5

# 测试曝光池刷新
python test_recommender.py --pool

# 保存测试结果到文件
python test_recommender.py --user 12345 --save
```

## API 接口

系统提供了以下 API 接口：

1. **获取推荐**：

   ```
   GET /api/recommendations?user_id=<user_id>&page=<page>&page_size=<page_size>
   ```

2. **获取热门话题**：

   ```
   GET /api/hot_topics?count=<count>
   ```

3. **标记内容为已浏览**：
   ```
   POST /api/mark_viewed
   Body: {"user_id": "<user_id>", "item_ids": [<item_id1>, <item_id2>, ...]}
   ```

### 示例：无限滚动推荐

前端实现无限滚动推荐的伪代码示例：

```javascript
let page = 1;
const pageSize = 20;
let isLoading = false;
let hasMore = true;
let viewedItems = new Set();

// 初始加载
loadRecommendations();

// 监听滚动事件
window.addEventListener("scroll", () => {
  if (isNearBottom() && !isLoading && hasMore) {
    loadMoreRecommendations();
  }
});

async function loadRecommendations() {
  isLoading = true;
  try {
    const response = await fetch(
      `/api/recommendations?user_id=${userId}&page=${page}&page_size=${pageSize}`
    );
    const data = await response.json();

    renderItems(data.recommendations);

    // 更新状态
    hasMore = data.has_more;
    page++;

    // 记录已浏览的内容，并告诉服务器
    const newItemIds = data.recommendations.map((item) => item.post_id);
    viewedItems = new Set([...viewedItems, ...newItemIds]);

    // 发送已浏览记录
    fetch("/api/mark_viewed", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: userId,
        item_ids: newItemIds,
      }),
    });
  } catch (error) {
    console.error("加载推荐失败:", error);
  } finally {
    isLoading = false;
  }
}

function loadMoreRecommendations() {
  loadRecommendations();
}
```

## 使用注意事项

1. **内存管理**：系统使用内存缓存，请根据服务器内存状况调整缓存大小
2. **多进程部署**：多进程部署时，缓存不共享，可能需要调整缓存策略
3. **数据库负载**：监控数据库负载，必要时对热点查询添加索引
4. **缓存过期设置**：根据数据更新频率调整缓存过期时间

## 性能优化建议

1. **数据库索引**：为常用查询添加合适的索引
2. **缓存预热**：启动时预加载热点数据到缓存
3. **参数调优**：根据实际情况调整连接池、缓存大小等参数
4. **批量操作**：使用批量查询替代多次单条查询

## 项目结构

```
recommend_system/
│
├── app.py                      # 主应用入口
├── run_api.sh                  # API服务启动脚本
├── test_recommender.py         # 推荐功能测试工具
├── README.md                   # 项目说明
├── schema_update.sql           # 数据库结构文件
├── requirements.txt            # 依赖列表
│
├── api/                        # API接口目录
│   └── routes.py               # API路由定义
│
├── config/                     # 配置目录
│   ├── default_config.yaml     # 默认配置文件
│   └── config.yaml             # 用户自定义配置(可选)
│
├── recommender/                # 推荐模块
│   └── engine.py               # 推荐引擎实现
│
├── exposure/                   # 曝光池模块
│   └── pool_manager.py         # 曝光池管理实现
│
├── hot_topics/                 # 热点话题模块
│   └── generator.py            # 热点话题生成实现
│
├── scheduler/                  # 任务调度模块
│   └── task_scheduler.py       # 任务调度器实现
│
├── cache/                      # 缓存模块
│   ├── cache_manager.py        # 缓存管理器
│   └── lru_cache.py            # LRU缓存实现
│
├── database/                   # 数据库模块
│   └── db_manager.py           # 数据库连接管理
│
├── utils/                      # 工具模块
│   ├── config_manager.py       # 配置加载工具
│   └── logger.py               # 日志工具
│
└── logs/                       # 日志目录
    └── ...                     # 日志文件
```

## 后续优化方向

1. **算法升级**：引入更先进的推荐算法，如深度学习模型
2. **A/B 测试框架**：构建测试框架，验证不同策略效果
3. **用户画像系统**：完善用户兴趣建模
4. **性能监控**：添加详细的性能监控指标
5. **分布式支持**：增强系统在分布式环境下的能力

## 总结

基于 Python 和 MySQL 的推荐系统提供了一个轻量级但功能完善的解决方案，无需依赖复杂的中间件。通过优化的内存管理、连接池和算法实现，系统可满足中小规模应用的推荐需求，同时保持部署简单、维护方便的特点。
