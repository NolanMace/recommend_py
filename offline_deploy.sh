#!/bin/bash

# 推荐系统离线部署脚本
# 用于在无网络环境下部署和运行推荐系统
# 作者：AI助手

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 显示标题
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}      推荐系统离线部署脚本                 ${NC}"
echo -e "${BLUE}============================================${NC}"

# 检查Python是否安装
command -v python >/dev/null 2>&1 || { 
    echo -e "${RED}错误: 未找到Python，请先安装Python 3.x${NC}" 
    exit 1
}

# 显示Python版本
echo -e "${GREEN}使用Python版本:${NC} $(python --version)"

# 检查虚拟环境目录
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}未找到虚拟环境目录 '${VENV_DIR}'${NC}"
    echo -e "${YELLOW}尝试使用打包的wheels目录安装依赖...${NC}"
    
    # 检查wheels目录
    if [ ! -d "wheels" ]; then
        echo -e "${RED}错误: 未找到wheels目录，无法离线安装依赖${NC}"
        echo -e "${RED}请确保项目包含预先下载的依赖包${NC}"
        exit 1
    fi
    
    # 创建新的虚拟环境
    echo -e "${GREEN}创建新的虚拟环境...${NC}"
    python -m venv "$VENV_DIR"
    
    # 激活虚拟环境
    echo -e "${GREEN}激活虚拟环境...${NC}"
    source "$VENV_DIR/bin/activate"
    
    # 升级pip
    echo -e "${GREEN}从wheels目录安装基础工具...${NC}"
    pip install --no-index --find-links=wheels pip setuptools wheel
    
    # 安装所有依赖
    echo -e "${GREEN}从wheels目录安装项目依赖...${NC}"
    pip install --no-index --find-links=wheels -r requirements.txt
else
    # 激活虚拟环境
    echo -e "${GREEN}使用现有虚拟环境...${NC}"
    source "$VENV_DIR/bin/activate"
fi

# 检查依赖是否安装成功
echo -e "${GREEN}检查关键依赖是否已安装...${NC}"
python -c "import numpy; print('NumPy版本:', numpy.__version__)" || {
    echo -e "${RED}依赖检查失败，可能安装不完整${NC}"
    exit 1
}

# 初始化数据库（如果需要）
if [[ "$*" == *"--init-db"* ]]; then
    echo -e "${YELLOW}初始化数据库...${NC}"
    python main.py --init-db
fi

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