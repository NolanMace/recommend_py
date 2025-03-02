#!/bin/bash

# 推荐系统环境设置脚本
# 作者：AI助手

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 显示标题
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}         推荐系统环境设置脚本               ${NC}"
echo -e "${BLUE}============================================${NC}"

# 检查Python版本
echo -e "${YELLOW}检查Python版本...${NC}"
python --version
if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 未找到Python，请先安装Python 3.9+${NC}"
    exit 1
fi

# 设置虚拟环境名称
VENV_NAME="venv"
echo -e "${YELLOW}将使用非隐藏的虚拟环境目录: ${VENV_NAME}${NC}"

# 创建虚拟环境
if [ -d "$VENV_NAME" ]; then
    echo -e "${YELLOW}发现已存在的虚拟环境，是否重新创建? [y/N]${NC}"
    read -p "" RECREATE
    if [[ $RECREATE == "y" || $RECREATE == "Y" ]]; then
        echo -e "${YELLOW}删除现有虚拟环境...${NC}"
        rm -rf "$VENV_NAME"
    else
        echo -e "${GREEN}将使用现有虚拟环境.${NC}"
    fi
fi

if [ ! -d "$VENV_NAME" ]; then
    echo -e "${GREEN}创建新的虚拟环境...${NC}"
    python -m venv "$VENV_NAME"
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 创建虚拟环境失败${NC}"
        exit 1
    fi
fi

# 激活虚拟环境
echo -e "${GREEN}激活虚拟环境...${NC}"
source "$VENV_NAME/bin/activate"
if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 激活虚拟环境失败${NC}"
    exit 1
fi

# 升级pip和setuptools
echo -e "${GREEN}升级pip和setuptools...${NC}"
pip install --upgrade pip setuptools wheel

# 安装依赖
echo -e "${GREEN}安装项目依赖...${NC}"
pip install -r requirements.txt

# 更新脚本中的虚拟环境引用
if grep -q "\.venv" run.sh; then
    echo -e "${YELLOW}更新run.sh脚本中的虚拟环境引用...${NC}"
    sed -i.bak "s/\.venv/$VENV_NAME/g" run.sh
    rm -f run.sh.bak
fi

# 修改执行权限
chmod +x run.sh

echo -e "${GREEN}环境设置完成!${NC}"
echo -e "${YELLOW}现在您可以通过以下命令运行项目:${NC}"
echo -e "./run.sh"
echo -e "${YELLOW}如果需要初始化数据库，请使用:${NC}"
echo -e "./run.sh --init-db"

# 退出虚拟环境
deactivate 