#!/bin/bash
# 推荐系统便携式Python环境激活脚本
export PORTABLE_PYTHON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/deps/portable_python"
export PATH="$PORTABLE_PYTHON_DIR/python-3.9.12/bin:$PATH"
export PYTHONHOME="$PORTABLE_PYTHON_DIR/python-3.9.12"
echo "已激活便携式Python环境"
echo "Python路径: $(which python)"
echo "Python版本: $(python --version)"
