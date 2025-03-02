# 推荐系统优化项目

## 项目概述

这是一个基于 Python 和 MySQL 的推荐系统，专注于高效、简洁的实现，无需依赖 Redis 或其他第三方服务。系统采用纯 Python 实现各种功能组件，包括内存缓存、数据库连接池和任务调度等。通过内存缓存优化和数据库连接池管理，提供高效的推荐服务，满足中小规模应用场景的需求。

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
- **ORM**: SQLAlchemy
- **任务调度**: APScheduler
- **Web 框架**: Flask (用于 API 服务)

## 主要功能模块

- **推荐引擎**: 结合多种算法的混合推荐引擎，包括协同过滤和内容特征推荐
- **曝光池管理**: 智能管理内容曝光机会，确保新内容、热门内容和优质内容的曝光比例
- **热点话题生成**: 动态识别和生成热点内容，基于用户行为和时间衰减算法
- **任务调度**: 优化的多优先级任务管理，支持定时任务和周期性任务
- **配置管理**: 基于 YAML 的集中式配置，支持默认配置和用户自定义配置
- **缓存管理**: 高效的内存 LRU 缓存策略，自动清理过期数据
- **数据库优化**: 连接池管理及查询优化，提供熔断机制保护数据库
- **无限滚动推荐**: 多层级推荐策略，支持用户无限向下刷新时提供持续推荐

## 无限滚动推荐策略

推荐系统实现了多层级的推荐策略，确保用户可以持续获取内容推荐：

1. **分层推荐**：

   - 前 3 页：严格个性化推荐
   - 4-10 页：混合推荐（个性化+热门内容）
   - 10 页以后：扩展推荐（降低相似度阈值，扩大兴趣范围）

2. **动态扩展**：随着滚动深度增加，系统会：

   - 逐步降低内容相似度阈值
   - 扩展用户兴趣领域
   - 增加推荐内容多样性

3. **兜底机制**：当推荐数量不足时，自动补充：
   - 热门内容
   - 最新内容
   - 高质量内容
4. **已浏览记录**：系统会记录用户已浏览内容，确保不会重复推荐。

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

## 系统特点

- **低依赖**: 仅依赖 Python 和 MySQL，无需 Redis 等外部缓存服务
- **高性能**: 通过内存缓存和连接池优化性能，减少数据库压力
- **高可靠**: 内置熔断器和异常处理机制，防止系统雪崩
- **可扩展**: 模块化设计，易于扩展和定制
- **易部署**: 简化的部署要求，适合各种环境
- **无限滚动**: 支持用户持续浏览，提供源源不断的内容

## 数据库连接池

系统使用优化的数据库连接池管理，主要特点：

- **连接复用**: 减少频繁创建和销毁连接的开销
- **自动重连**: 连接失效时自动重新连接
- **超时控制**: 防止长时间占用连接资源
- **熔断保护**: 当数据库连接频繁失败时，暂时阻断请求
- **支持 ORM**: 同时提供原生 SQL 和 ORM 接口

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
   ```

4. 配置数据库连接
   创建 `config/config.yaml` 文件（可选，系统会使用默认配置）

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

    // 记录已浏览的内容
    data.recommendations.forEach((item) => viewedItems.add(item.post_id));
  } catch (error) {
    console.error("加载推荐失败:", error);
  } finally {
    isLoading = false;
  }
}

function loadMoreRecommendations() {
  loadRecommendations();
}

function isNearBottom() {
  return (
    window.innerHeight + window.scrollY >= document.body.offsetHeight - 500
  );
}

function renderItems(items) {
  // 渲染推荐内容到页面
  const container = document.getElementById("recommendations-container");
  items.forEach((item) => {
    const element = createItemElement(item);
    container.appendChild(element);
  });
}
```

## 配置文件

系统使用 YAML 格式的配置文件，默认位置为 `config/default_config.yaml`。用户可以创建 `config/config.yaml` 覆盖默认配置。配置项包括：

- 数据库连接参数
- 缓存策略设置
- 推荐算法权重
- 曝光池容量和权重
- 任务调度频率和优先级
- 日志级别和输出方式
- 熔断器参数
- 分页加载设置
- 无限滚动策略参数

## 使用注意事项

1. **内存管理**

   - 由于使用内存缓存替代 Redis，请根据实际负载监控内存使用情况
   - 对于大规模应用，建议调整`max_size`参数限制缓存大小

2. **多进程部署**

   - 内存缓存不支持跨进程共享，多进程部署时每个进程会维护独立的缓存
   - 如需多进程共享缓存，考虑引入共享内存方案或恢复 Redis 支持

3. **数据库负载**

   - 监控数据库连接使用情况，必要时调整连接池大小
   - 为频繁查询的表添加适当的索引

4. **缓存过期时间**
   - 根据数据更新频率调整缓存过期时间
   - 热点数据建议较短过期时间，静态数据可设置较长过期时间

## 系统要求

- Python 3.8+
- MySQL 5.7+
- 至少 1GB 可用内存 (取决于配置的缓存大小)

## 项目结构

```
recommend_py/
├── app.py                  # 主应用入口
├── config/                 # 配置管理
│   ├── config_manager.py   # 配置管理器
│   ├── default_config.yaml # 默认配置文件
├── cache/                  # 缓存管理
│   ├── cache_manager.py    # 缓存管理器
│   ├── lru_cache.py        # LRU缓存实现
├── database/               # 数据库管理
│   ├── db_pool.py          # 数据库连接池
├── recommender/            # 推荐引擎
│   ├── engine.py           # 推荐引擎主类
│   ├── algorithms/         # 推荐算法实现
├── exposure/               # 曝光池管理
│   ├── pool_manager.py     # 曝光池管理器
├── hot_topics/             # 热点话题生成
│   ├── generator.py        # 热点话题生成器
├── scheduler/              # 任务调度
│   ├── task_scheduler.py   # 任务调度器
├── models/                 # 模型存储目录
└── logs/                   # 日志文件目录
```

## 性能优化

为获得最佳性能，推荐以下优化措施：

1. **数据库索引优化**

   - 为用户 ID、内容 ID 等频繁查询的字段创建索引
   - 考虑使用复合索引优化复杂查询

2. **缓存预热**

   - 系统启动时主动加载热门内容到缓存
   - 定期预生成活跃用户的推荐结果

3. **参数调优**

   - 根据硬件资源调整数据库连接池大小
   - 根据内存容量调整缓存大小
   - 根据 CPU 核心数调整任务线程数

4. **批量操作**
   - 使用批量查询和更新替代单条操作
   - 使用批量插入减少数据库操作次数
