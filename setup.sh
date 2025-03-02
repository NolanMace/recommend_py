#!/bin/bash

# 推荐系统安装脚本 - 联网环境版本
# 用于创建虚拟环境和安装所有必要依赖

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 显示标题
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}      推荐系统安装脚本（联网版本）         ${NC}"
echo -e "${BLUE}============================================${NC}"

# 设置虚拟环境目录
VENV_DIR="venv"

# 检查是否需要重建虚拟环境
if [ "$1" == "--force-recreate" ]; then
    echo -e "${YELLOW}将强制重建虚拟环境${NC}"
    if [ -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}删除现有虚拟环境...${NC}"
        rm -rf "$VENV_DIR"
    fi
fi

# 检查Python版本
python_version=$(python3 --version 2>/dev/null || python --version)
echo -e "${GREEN}系统Python版本: $python_version${NC}"

# 确定Python命令
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# 检查虚拟环境
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${GREEN}创建新的虚拟环境...${NC}"
    $PYTHON_CMD -m venv "$VENV_DIR"
else
    echo -e "${GREEN}使用现有虚拟环境...${NC}"
fi

# 激活虚拟环境
echo -e "${GREEN}激活虚拟环境...${NC}"
source "$VENV_DIR/bin/activate"

# 升级pip
echo -e "${GREEN}升级pip...${NC}"
pip install --upgrade pip setuptools wheel

# 安装依赖
echo -e "${GREEN}安装项目依赖...${NC}"
pip install -r requirements.txt

# 检查关键依赖是否安装成功
echo -e "${GREEN}检查关键依赖...${NC}"
python -c "import numpy; print(f'NumPy版本: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas版本: {pandas.__version__}')"
python -c "import schedule; print(f'Schedule版本: {schedule.__version__}')"

echo -e "${GREEN}安装完成!${NC}"
echo -e "${YELLOW}使用方法:${NC}"
echo -e "1. 直接运行: ./run.sh"
echo -e "2. 初始化数据库: ./run.sh --init-db"
echo -e "3. 调试模式: ./run.sh --debug"
echo -e "4. 执行测试任务: ./run.sh --run-tasks"
echo -e "5. 禁用调度器: ./run.sh --no-scheduler"

# 退出虚拟环境
deactivate 