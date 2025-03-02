#!/bin/bash

# API服务启动脚本

# 定义颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 当前时间
TIME=$(date +"%Y-%m-%d %H:%M:%S")

# 可能的Python命令
PYTHON_COMMANDS=("python3" "python" "py")

# 检查虚拟环境
if [ -d "venv" ]; then
    echo -e "${BLUE}[${TIME}] 发现虚拟环境，正在激活...${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}[${TIME}] 未找到虚拟环境，尝试使用系统Python...${NC}"
fi

# 查找可用的Python命令
PYTHON_CMD=""
for cmd in "${PYTHON_COMMANDS[@]}"; do
    if command -v $cmd &> /dev/null; then
        PYTHON_CMD=$cmd
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}[${TIME}] 错误: 找不到可用的Python命令${NC}"
    exit 1
fi

echo -e "${BLUE}[${TIME}] 使用Python命令: ${PYTHON_CMD}${NC}"

# 启动API服务
echo -e "${GREEN}[${TIME}] 启动API服务...${NC}"

# 解析命令行参数
DEBUG=""
HOST="0.0.0.0"
PORT="5000"

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG="--debug"
            shift
            ;;
        --host=*)
            HOST="${1#*=}"
            shift
            ;;
        --port=*)
            PORT="${1#*=}"
            shift
            ;;
        *)
            echo -e "${YELLOW}[${TIME}] 警告: 未知参数 $1${NC}"
            shift
            ;;
    esac
done

# 创建日志目录
mkdir -p logs

# 启动API服务
LOG_FILE="logs/api_$(date +"%Y%m%d_%H%M%S").log"
echo -e "${BLUE}[${TIME}] 日志将记录到: ${LOG_FILE}${NC}"

${PYTHON_CMD} api/api.py --host=${HOST} --port=${PORT} ${DEBUG} 2>&1 | tee ${LOG_FILE}

# 捕获Ctrl+C
trap ctrl_c INT
function ctrl_c() {
    echo -e "${YELLOW}[$(date +"%Y-%m-%d %H:%M:%S")] 捕获到中断信号，正在退出...${NC}"
    exit 0
}

echo -e "${GREEN}[$(date +"%Y-%m-%d %H:%M:%S")] API服务已启动，监听: ${HOST}:${PORT}${NC}" 