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

# 检查便携式Python是否存在
PORTABLE_PYTHON="/root/deps/portable_python/python-3.9.12/bin/python"
if [ ! -f "$PORTABLE_PYTHON" ]; then
    echo -e "${RED}错误: 未找到便携式Python${NC}"
    echo -e "${RED}请确保已运行prepare_portable_python.sh脚本${NC}"
    exit 1
fi

# 显示Python版本
echo -e "${GREEN}使用便携式Python版本:${NC} $($PORTABLE_PYTHON --version)"

# 设置虚拟环境目录
VENV_DIR="venv"

# 强制重新创建虚拟环境
if [ "$1" == "--force-recreate-venv" ] || [ "$2" == "--force-recreate-venv" ] || [ "$3" == "--force-recreate-venv" ] || [ "$4" == "--force-recreate-venv" ]; then
    echo -e "${YELLOW}参数指定强制重新创建虚拟环境${NC}"
    if [ -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}删除现有虚拟环境...${NC}"
        rm -rf "$VENV_DIR"
    fi
fi

# 检查deps/wheels目录
if [ ! -d "deps/wheels" ]; then
    echo -e "${RED}错误: 未找到deps/wheels目录，无法离线安装依赖${NC}"
    echo -e "${RED}请确保项目包含预先下载的依赖包${NC}"
    exit 1
fi

# 检查虚拟环境目录
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}未找到虚拟环境目录 '${VENV_DIR}'，将创建新的虚拟环境${NC}"
else
    # 检查虚拟环境是否完好
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        echo -e "${YELLOW}虚拟环境似乎不完整，将重新创建${NC}"
        rm -rf "$VENV_DIR"
    else
        echo -e "${YELLOW}发现现有虚拟环境，但可能不兼容当前系统${NC}"
        echo -e "${YELLOW}建议使用 --force-recreate-venv 参数重新创建虚拟环境${NC}"
        echo -e "${YELLOW}例如: ./offline_deploy.sh --force-recreate-venv${NC}"
        echo -e "${YELLOW}您希望现在重新创建虚拟环境吗? [Y/n]${NC}"
        read -p "" RECREATE
        if [[ $RECREATE != "n" && $RECREATE != "N" ]]; then
            echo -e "${YELLOW}删除现有虚拟环境...${NC}"
            rm -rf "$VENV_DIR"
        fi
    fi
fi

# 创建新的虚拟环境（如果需要）
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${GREEN}创建新的虚拟环境...${NC}"
    "$PORTABLE_PYTHON" -m venv "$VENV_DIR"
    
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${RED}错误: 无法创建虚拟环境${NC}"
        exit 1
    fi
fi

# 激活虚拟环境
echo -e "${GREEN}激活虚拟环境...${NC}"
source "$VENV_DIR/bin/activate"

# 确认虚拟环境已激活
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}错误: 虚拟环境激活失败${NC}"
    exit 1
fi

echo -e "${GREEN}虚拟环境激活成功: $VIRTUAL_ENV${NC}"

# 升级pip
echo -e "${GREEN}从deps/wheels目录安装基础工具...${NC}"
"$VENV_DIR/bin/pip" install --upgrade --no-index --find-links=deps/wheels pip setuptools wheel

# 安装所有依赖
echo -e "${GREEN}从deps/wheels目录安装项目依赖...${NC}"
"$VENV_DIR/bin/pip" install --no-index --find-links=deps/wheels -r requirements.txt

# 检查依赖是否安装成功
echo -e "${GREEN}检查关键依赖是否已安装...${NC}"
"$VENV_DIR/bin/python" -c "import numpy; print('NumPy版本:', numpy.__version__)" || {
    echo -e "${YELLOW}未找到NumPy，尝试单独安装...${NC}"
    "$VENV_DIR/bin/pip" install --no-index --find-links=deps/wheels numpy
}

"$VENV_DIR/bin/python" -c "import schedule; print('Schedule版本:', schedule.__version__)" || {
    echo -e "${YELLOW}未找到Schedule，尝试单独安装...${NC}"
    "$VENV_DIR/bin/pip" install --no-index --find-links=deps/wheels schedule
}

"$VENV_DIR/bin/python" -c "import pandas; print('Pandas版本:', pandas.__version__)" || {
    echo -e "${YELLOW}未找到Pandas，尝试单独安装...${NC}"
    "$VENV_DIR/bin/pip" install --no-index --find-links=deps/wheels pandas
}

# 确认关键依赖安装成功
DEPS_OK=true
"$VENV_DIR/bin/python" -c "import numpy" || DEPS_OK=false
"$VENV_DIR/bin/python" -c "import schedule" || DEPS_OK=false

if [ "$DEPS_OK" = false ]; then
    echo -e "${RED}错误: 一些关键依赖安装失败${NC}"
    echo -e "${YELLOW}请检查deps/wheels目录是否包含所有必要的依赖包${NC}"
    echo -e "${YELLOW}您可以继续尝试运行应用，但可能会出现错误${NC}"
    echo -e "${YELLOW}是否继续? [y/N]${NC}"
    read -p "" CONTINUE
    if [[ $CONTINUE != "y" && $CONTINUE != "Y" ]]; then
        echo -e "${RED}操作已取消${NC}"
        deactivate
        exit 1
    fi
fi

# 初始化数据库（如果需要）
if [[ "$*" == *"--init-db"* ]]; then
    echo -e "${YELLOW}初始化数据库...${NC}"
    "$VENV_DIR/bin/python" main.py --init-db
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
    "$VENV_DIR/bin/python" main.py $RUN_ARGS
else
    echo -e "${YELLOW}参数: $RUN_ARGS --log-file $LOG_FILE${NC}"
    echo -e "${GREEN}日志将保存到: $LOG_FILE${NC}"
    "$VENV_DIR/bin/python" main.py $RUN_ARGS --log-file "$LOG_FILE"
    echo -e "${GREEN}系统已退出，查看完整日志:${NC}"
    echo -e "cat $LOG_FILE"
fi

# 退出虚拟环境
deactivate 