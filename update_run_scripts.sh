#!/bin/bash

# 用于更新运行脚本中虚拟环境引用的脚本
# 作者: AI助手

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 显示标题
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}    更新运行脚本中的虚拟环境引用           ${NC}"
echo -e "${BLUE}============================================${NC}"

# 检查参数
if [ "$#" -ne 2 ]; then
    echo -e "${RED}使用方法: $0 <旧的虚拟环境名称> <新的虚拟环境名称>${NC}"
    echo -e "${YELLOW}例如: $0 .venv venv${NC}"
    exit 1
fi

OLD_ENV="$1"
NEW_ENV="$2"

echo -e "${YELLOW}将把脚本中的 '$OLD_ENV' 替换为 '$NEW_ENV'${NC}"

# 更新run.sh
if [ -f "run.sh" ]; then
    echo -e "${GREEN}更新run.sh...${NC}"
    sed -i.bak "s/$OLD_ENV/$NEW_ENV/g" run.sh
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}run.sh 更新成功${NC}"
        rm -f run.sh.bak
    else
        echo -e "${RED}run.sh 更新失败${NC}"
    fi
else
    echo -e "${YELLOW}未找到 run.sh${NC}"
fi

# 更新run.bat
if [ -f "run.bat" ]; then
    echo -e "${GREEN}更新run.bat...${NC}"
    sed -i.bak "s/$OLD_ENV/$NEW_ENV/g" run.bat
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}run.bat 更新成功${NC}"
        rm -f run.bat.bak
    else
        echo -e "${RED}run.bat 更新失败${NC}"
    fi
else
    echo -e "${YELLOW}未找到 run.bat${NC}"
fi

echo -e "${GREEN}脚本引用更新完成!${NC}"
echo -e "${YELLOW}请确认脚本中的虚拟环境引用已更新为 '$NEW_ENV'${NC}" 