# 帖子推荐系统

## 系统功能

本系统是一个完整的帖子推荐引擎，主要功能包括：

1. **个性化帖子推荐**：基于用户历史行为和兴趣特征，将推荐结果直接写入数据库
2. **曝光池机制**：设计分层曝光池策略，每次帖子有推送时间和推送次数限制
3. **热点话题生成**：每 5 分钟自动生成 50 个热点话题，结果存入数据库
4. **批量推荐生成**：定时为活跃用户批量生成推荐结果，并写入 MySQL 表

## 系统架构

系统由以下几个主要模块组成：

- **推荐算法引擎**：基于 TF-IDF 的内容推荐算法，结合用户行为权重
- **曝光池管理器**：多级曝光池策略，控制帖子推送频率和范围
- **热点生成器**：定时计算帖子热度，生成热点话题列表
- **定时任务调度器**：管理模型更新、热点生成、批量推荐等定时任务
- **数据库存储模块**：将推荐结果和热点数据持久化到 MySQL

## 数据库设计

系统基于 MySQL 数据库，主要表结构包括：

- `posts`：帖子表，新增曝光池级别、热度分数、曝光次数等字段
- `user_behavior`：用户行为表，记录浏览、点赞、收藏、评论等行为
- `user_search_behavior`：用户搜索行为表，记录搜索词和时间
- `post_exposures`：帖子曝光记录表，记录推送历史
- `hot_topics_history`：热点话题历史表，记录生成的热点话题
- `user_recommendations`：用户推荐结果表，保存生成的推荐结果
- `current_hot_topics`：当前热点表，保存最新的热点数据

## 曝光池机制

曝光池是控制帖子曝光机会的机制，分为三个层级：

1. **第一曝光池**：低热度新帖，占曝光池推荐 50%
2. **第二曝光池**：中热度帖子，占曝光池推荐 30%
3. **第三曝光池**：高热度帖子，占曝光池推荐 20%

每个帖子根据热度得分自动进入不同的曝光池，并在达到一定热度时升级到更高级的曝光池。

## 热点生成机制

系统每 5 分钟自动计算一次帖子热度，生成 50 个热点话题。热度计算基于多种用户行为：

- 浏览：权重 1
- 点赞：权重 3
- 收藏：权重 4
- 评论：权重 5

## 数据库存储机制

系统将数据写入以下表：

1. **用户推荐结果表** (`user_recommendations`)

   - 每个用户的推荐结果以 JSON 格式保存
   - 设置过期时间自动失效
   - 默认最多保留每用户最近 3 次的推荐结果

2. **当前热点表** (`current_hot_topics`)

   - 保存最新一次生成的热点帖子
   - 包含排名位置、热度分数、帖子标题等信息

3. **历史记录表**
   - 保存热点历史记录和曝光历史
   - 设置自动清理机制避免数据过多

## 定时任务

系统设置了以下定时任务：

1. **模型更新**：每天凌晨 00:30 更新 TF-IDF 模型
2. **热点生成**：每 5 分钟生成一次热点话题
3. **批量推荐**：每 60 分钟为活跃用户批量生成推荐结果
4. **数据清理**：每天凌晨 02:00 清理过期数据

## 部署与启动

### 常规部署（联网环境）

#### 安装依赖

```bash
pip install -r requirements.txt
```

#### 启动服务

```bash
python main.py --log-file ./recommend_system.log
```

### 离线部署（无网络环境）

本系统提供完整的离线部署方案，适用于无法联网的服务器环境。

#### 准备离线部署包（在联网环境执行）

1. 下载所有依赖并准备部署包：

```bash
# 设置执行权限
chmod +x prepare_offline_package.sh

# 执行打包脚本
./prepare_offline_package.sh
```

2. 将整个项目目录打包传输到离线服务器：

```bash
# Linux/macOS
tar -czf recommend_system.tar.gz 项目目录/

# 或Windows环境
zip -r recommend_system.zip 项目目录/
```

#### 离线部署步骤

##### Linux/macOS 环境

1. 解压部署包：

```bash
tar -xzf recommend_system.tar.gz
cd 项目目录
```

2. 运行离线部署脚本：

```bash
# 确保脚本有执行权限
chmod +x offline_deploy.sh

# 首次部署（初始化数据库）
./offline_deploy.sh --init-db

# 常规运行
./offline_deploy.sh
```

3. 其他运行选项：

```bash
# 调试模式运行
./offline_deploy.sh --debug

# 禁用调度器运行
./offline_deploy.sh --no-scheduler

# 不保存日志运行
./offline_deploy.sh --no-log

# 组合多个参数
./offline_deploy.sh --init-db --debug
```

### 便携式 Python 部署（无需安装 Python）

本系统支持完全便携式部署，即使在目标服务器没有安装 Python 的情况下也能运行。系统会携带一个完整的 Python 解释器和所有依赖包。

#### 准备便携式部署包（在联网环境执行）

1. 下载 Python 解释器并准备便携式部署包：

```bash
# 设置执行权限
chmod +x prepare_portable_python.sh

# 执行便携式Python准备脚本
./prepare_portable_python.sh
```

该脚本会：

- 下载并编译 Python 3.9.12 解释器
- 创建 deps/portable_python 目录存放 Python 解释器
- 下载所有依赖到 deps/wheels 目录
- 更新部署脚本以使用便携式 Python

2. 将整个项目目录（包括便携式 Python）打包传输到离线服务器：

```bash
# Linux/macOS
tar -czf recommend_system_portable.tar.gz 项目目录/

# 或Windows环境
zip -r recommend_system_portable.zip 项目目录/
```

#### 便携式部署步骤

1. 解压部署包：

```bash
tar -xzf recommend_system_portable.tar.gz
cd 项目目录
```

2. 运行离线部署脚本（与标准部署相同）：

```bash
# 确保脚本有执行权限
chmod +x offline_deploy.sh

# 首次部署（初始化数据库）
./offline_deploy.sh --init-db

# 常规运行
./offline_deploy.sh
```

3. 如果遇到运行问题，强制重建虚拟环境：

```bash
./offline_deploy.sh --force-recreate-venv
```

#### 便携式部署的优势

- **零依赖**：目标服务器无需安装 Python 或任何其他软件
- **版本一致性**：保证所有服务器使用完全相同的 Python 版本和依赖
- **完全隔离**：避免与服务器上已有的 Python 环境冲突
- **自包含**：整个系统作为独立单元，可在任何 Linux/macOS 服务器上运行
