#!/bin/bash

echo "==========================================="
echo "      推荐系统安装脚本（优化版本）         "
echo "==========================================="

# 检查Python版本
python_version=$(python3 -V 2>&1)
echo "系统Python版本: $python_version"

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "创建新的虚拟环境..."
    python3 -m venv venv
else
    echo "使用现有虚拟环境..."
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 升级pip并安装基础工具
echo "升级pip和基础工具..."
python -m pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120

# 安装项目依赖
echo "安装项目依赖..."
# 首先安装numpy（因为它是基础依赖）
pip install numpy==1.24.3 -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120

# 安装其他依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120

# 检查关键依赖
echo "检查关键依赖..."
python -c "import numpy; print(f'NumPy版本: {numpy.__version__}')" 2>/dev/null || echo "警告: NumPy 未正确安装"
python -c "import pandas; print(f'Pandas版本: {pandas.__version__}')" 2>/dev/null || echo "警告: Pandas 未正确安装"
python -c "import schedule; print(f'Schedule版本: {schedule.__version__}')" 2>/dev/null || echo "警告: Schedule 未正确安装"

echo "安装完成!"
echo "使用方法:"
echo "1. 直接运行: ./run.sh"
echo "2. 初始化数据库: ./run.sh --init-db"
echo "3. 调试模式: ./run.sh --debug"
echo "4. 执行测试任务: ./run.sh --run-tasks"
echo "5. 禁用调度器: ./run.sh --no-scheduler" 