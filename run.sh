#!/bin/bash

# 推荐系统运行脚本
# 用于运行推荐系统应用

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 显示标题
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}      推荐系统运行脚本                     ${NC}"
echo -e "${BLUE}============================================${NC}"

# 设置虚拟环境目录
VENV_DIR="venv"

# 检查虚拟环境
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}未找到虚拟环境，正在设置...${NC}"
    ./setup.sh
    
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${RED}错误: 无法创建虚拟环境${NC}"
        exit 1
    fi
fi

# 激活虚拟环境
echo -e "${GREEN}激活虚拟环境...${NC}"
source "$VENV_DIR/bin/activate"

# 运行应用
echo -e "${GREEN}启动推荐系统...${NC}"

# 准备参数
RUN_ARGS=""
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/recommend_system_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

# 初始化数据库（如果需要）
if [[ "$*" == *"--init-db"* ]]; then
    echo -e "${YELLOW}初始化数据库...${NC}"
    python main.py --init-db
    RUN_ARGS="$RUN_ARGS --init-db"
fi

# 检查参数
if [[ "$*" == *"--debug"* ]]; then
    RUN_ARGS="$RUN_ARGS --debug"
fi

if [[ "$*" == *"--no-scheduler"* ]]; then
    RUN_ARGS="$RUN_ARGS --no-scheduler"
fi

if [[ "$*" == *"--no-log"* ]]; then
    echo -e "${YELLOW}参数: $RUN_ARGS${NC}"
    python main.py $RUN_ARGS
else
    echo -e "${YELLOW}参数: $RUN_ARGS --log-file $LOG_FILE${NC}"
    echo -e "${GREEN}日志将保存到: $LOG_FILE${NC}"
    python main.py $RUN_ARGS --log-file "$LOG_FILE"
    echo -e "${GREEN}系统已退出，查看完整日志:${NC}"
    echo -e "cat $LOG_FILE"
fi

# 退出虚拟环境
deactivate 