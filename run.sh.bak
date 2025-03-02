#!/bin/bash

# 推荐系统启动脚本 (macOS/Linux)
# 作者：AI助手
# 创建日期：$(date)

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 显示标题
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}         推荐系统启动脚本 (macOS/Linux)        ${NC}"
echo -e "${BLUE}============================================${NC}"

# 检测虚拟环境
if [ ! -d ".venv" ]; then
    echo -e "${RED}错误: 未找到虚拟环境 (.venv 目录)${NC}"
    echo -e "${YELLOW}请先创建虚拟环境：${NC}"
    echo -e "python -m venv .venv"
    echo -e "source .venv/bin/activate"
    echo -e "pip install -r requirements.txt"
    exit 1
fi

# 获取命令行参数
ARGS="$@"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/recommend_system_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录
if [ ! -d "$LOG_DIR" ]; then
    echo -e "${YELLOW}创建日志目录: $LOG_DIR${NC}"
    mkdir -p "$LOG_DIR"
fi

# 激活虚拟环境
echo -e "${GREEN}正在激活虚拟环境...${NC}"
source .venv/bin/activate

# 检查虚拟环境是否激活成功
if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 无法激活虚拟环境${NC}"
    exit 1
fi

# 显示Python版本
echo -e "${GREEN}使用Python版本:${NC} $(python --version)"

# 帮助信息
function show_help() {
    echo -e "${YELLOW}使用方法:${NC}"
    echo -e "  ./run.sh [选项]"
    echo -e ""
    echo -e "${YELLOW}可用选项:${NC}"
    echo -e "  --help          显示此帮助信息"
    echo -e "  --init-db       初始化数据库表结构"
    echo -e "  --debug         启用调试模式"
    echo -e "  --no-scheduler  禁用调度任务服务"
    echo -e "  --no-log        不保存日志文件"
    echo -e ""
    echo -e "${YELLOW}示例:${NC}"
    echo -e "  ./run.sh --init-db      # 初始化数据库并启动系统"
    echo -e "  ./run.sh --debug        # 以调试模式启动系统"
}

# 处理帮助参数
if [[ "$ARGS" == *"--help"* ]]; then
    show_help
    exit 0
fi

# 准备启动参数
RUN_ARGS=""
if [[ "$ARGS" == *"--init-db"* ]]; then
    RUN_ARGS="$RUN_ARGS --init-db"
fi

if [[ "$ARGS" == *"--debug"* ]]; then
    RUN_ARGS="$RUN_ARGS --debug"
fi

if [[ "$ARGS" == *"--no-scheduler"* ]]; then
    RUN_ARGS="$RUN_ARGS --no-scheduler"
fi

# 判断是否记录日志
if [[ "$ARGS" == *"--no-log"* ]]; then
    echo -e "${GREEN}启动推荐系统...${NC}"
    echo -e "${YELLOW}参数: $RUN_ARGS${NC}"
    python main.py $RUN_ARGS
else
    echo -e "${GREEN}启动推荐系统...${NC}"
    echo -e "${YELLOW}参数: $RUN_ARGS --log-file $LOG_FILE${NC}"
    echo -e "${GREEN}日志将保存到: $LOG_FILE${NC}"
    python main.py $RUN_ARGS --log-file "$LOG_FILE"
    
    # 系统退出后显示日志路径
    echo -e "${GREEN}系统已退出，查看完整日志:${NC}"
    echo -e "cat $LOG_FILE"
fi

# 退出虚拟环境
deactivate 