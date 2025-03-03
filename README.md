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
- **任务调度**: Schedule
- **机器学习**: scikit-learn
- **模型存储**: joblib

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

### 系统要求

- **Python**: 3.8+
- **MySQL**: 5.7+

### 依赖安装

```bash
# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安装基础依赖
python3 -m pip install numpy pandas schedule pymysql scikit-learn joblib -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装其他项目依赖
python3 -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 配置说明

系统使用 YAML 格式的配置文件，位于 `config` 目录下：

- `config/default_config.yaml`: 默认配置文件
- `config/config.yaml`: 用户自定义配置（优先级更高）

数据库配置示例：

```yaml
database:
  mysql:
    host: localhost
    port: 3306
    user: root
    password: root
    database: recommend_system
    charset: utf8mb4
    pool_size: 10
    pool_recycle: 3600
    pool_timeout: 30
    connect_timeout: 10

circuit_breaker:
  enabled: true
  services:
    database:
      failure_threshold: 3
      recovery_timeout: 60
```

所有配置都可以通过配置管理器动态访问：

```python
from config.config_manager import get_config_manager

config = get_config_manager()
db_config = config.get('database', {}).get('mysql', {})
```

### 启动服务

```bash
# 初始化数据库（首次运行需要）
python3 main.py --init-db

# 启动服务（默认模式）
python3 main.py

# 调试模式启动（输出详细日志）
python3 main.py --debug

# 执行测试任务（立即执行所有定时任务）
python3 main.py --run-tasks

# 禁用调度器启动（不运行定时任务）
python3 main.py --no-scheduler

# 指定日志文件路径（默认输出到 logs/app.log）
python3 main.py --log-file logs/custom.log
```

## 命令行参数说明

- `--init-db`: 初始化数据库表结构和基础数据（首次运行时需要）
- `--debug`: 启用调试模式，输出更详细的日志信息，方便开发调试
- `--run-tasks`: 立即执行所有定时任务（热门内容更新、用户兴趣模型更新、全量数据分析）
- `--no-scheduler`: 禁用定时任务调度器，适用于只需要 API 服务的场景
- `--log-file PATH`: 指定日志文件路径，默认输出到 logs/app.log

### 日志级别说明

- DEBUG: 详细的调试信息，包括 SQL 查询和缓存操作
- INFO: 常规操作信息，如任务执行、服务启动
- WARNING: 需要注意但不影响系统运行的问题
- ERROR: 影响功能但不影响系统稳定性的错误
- CRITICAL: 严重错误，可能导致系统不可用

## 测试指南

### 1. 环境准备

```bash
# 创建并激活虚拟环境
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# 或
.\venv\Scripts\activate  # Windows

# 安装测试依赖
python3 -m pip install pytest pytest-cov pytest-benchmark -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 导入测试数据

```bash
# 导入测试数据集
python3 tests/import_test_data.py
```

测试数据包含：

- `tests/data/test_posts.csv`: 1000 条测试帖子
- `tests/data/test_users.csv`: 100 个测试用户
- `tests/data/test_interactions.csv`: 1000 条用户交互记录

### 3. 运行测试

```bash
# 运行所有测试并生成覆盖率报告
python3 -m pytest tests/ --cov=. --cov-report=html

# 只运行特定模块的测试
python3 -m pytest tests/test_recommender.py  # 推荐系统测试
python3 -m pytest tests/test_cache.py        # 缓存系统测试
python3 -m pytest tests/test_db_pool.py      # 数据库连接池测试

# 运行性能测试
python3 -m pytest tests/test_performance.py --benchmark-only

# 运行集成测试
python3 -m pytest tests/test_integration.py
```

### 4. 测试用例说明

#### 单元测试

- **缓存测试** (`test_cache.py`)

  - LRU 策略验证
  - 过期清理测试
  - 线程安全测试
  - 内存限制测试

- **数据库连接池测试** (`test_db_pool.py`)

  - 连接获取和释放
  - 自动重连机制
  - 连接池满载测试
  - 熔断器功能测试

- **推荐算法测试** (`test_recommender.py`)
  - 个性化推荐准确性
  - 冷启动处理
  - 推荐多样性
  - 实时性测试

#### 集成测试 (`test_integration.py`)

- 完整推荐流程测试
- 定时任务联动测试
- 配置热更新测试
- 异常恢复测试

#### 性能测试 (`test_performance.py`)

- 并发推荐请求测试（100/500/1000 并发）
- 缓存命中率测试
- 数据库连接池压力测试
- 内存使用监控

### 5. 测试报告查看

```bash
# 查看HTML格式的覆盖率报告
open htmlcov/index.html  # Mac
# 或
start htmlcov/index.html  # Windows

# 查看性能测试报告
python3 -m pytest tests/test_performance.py --benchmark-only --benchmark-histogram
```

### 6. 常见问题排查

1. 数据库连接失败

```bash
# 检查MySQL服务状态
mysql.server status  # Mac
# 或
net start mysql     # Windows
```

2. 测试数据导入失败

```bash
# 检查数据文件权限
ls -l tests/data/

# 手动执行SQL导入
mysql -u root -p recommend_system < tests/data/init.sql
```

3. 测试执行超时

```bash
# 使用-v参数查看详细日志
python3 -m pytest tests/ -v

# 单独执行超时的测试用例
python3 -m pytest tests/test_performance.py::test_concurrent_requests -v
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
├── main.py                      # 主应用入口
├── README.md                    # 项目说明文档
├── requirements.txt             # 项目依赖列表
│
├── api/                         # API接口模块
│   └── routes.py               # API路由定义
│
├── cache/                       # 缓存模块
│   └── cache.py                # LRU缓存实现
│
├── config/                      # 配置模块
│   ├── config.py               # 配置管理器
│   ├── default_config.yaml     # 默认配置文件
│   └── config.yaml             # 用户自定义配置
│
├── database/                    # 数据库模块
│   ├── database.py             # 数据库管理器
│   ├── migrate_to_db.py        # 数据库迁移工具
│   └── schema_update.sql       # 数据库结构定义
│
├── exposure/                    # 曝光池模块
│   └── pool_manager.py         # 曝光池管理实现
│
├── hot_topics/                  # 热点话题模块
│   └── generator.py            # 热点话题生成器
│
├── recommender/                 # 推荐引擎模块
│   └── recommender.py          # 推荐算法实现
│
├── scheduler/                   # 任务调度模块
│   └── scheduler.py            # 任务调度器实现
│
├── scripts/                     # 脚本工具目录
│   └── upline.sh               # 部署脚本
│
├── tests/                      # 测试目录
│   ├── test_recommender.py     # 推荐功能测试
│   └── test_infinite_scroll.py # 无限滚动测试
│
├── utils/                      # 工具模块
│   └── monitor.py             # 系统监控工具
│
└── logs/                       # 日志目录
    └── *.log                   # 日志文件
```

### 模块说明

- **api**: 提供 RESTful API 接口，处理外部请求
- **cache**: 实现基于 LRU 的内存缓存系统
- **config**: 管理系统配置和环境设置
- **database**: 处理数据库连接和数据访问
- **exposure**: 管理内容曝光和展示策略
- **hot_topics**: 处理热点话题的发现和管理
- **recommender**: 核心推荐算法实现
- **scheduler**: 管理定时任务和调度
- **scripts**: 部署和维护脚本
- **tests**: 单元测试和集成测试
- **utils**: 通用工具和监控组件

## 后续优化方向

1. **算法升级**：引入更先进的推荐算法，如深度学习模型
2. **A/B 测试框架**：构建测试框架，验证不同策略效果
3. **用户画像系统**：完善用户兴趣建模
4. **性能监控**：添加详细的性能监控指标
5. **分布式支持**：增强系统在分布式环境下的能力

## 总结

基于 Python 和 MySQL 的推荐系统提供了一个轻量级但功能完善的解决方案，无需依赖复杂的中间件。通过优化的内存管理、连接池和算法实现，系统可满足中小规模应用的推荐需求，同时保持部署简单、维护方便的特点。
