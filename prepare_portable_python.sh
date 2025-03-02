#!/bin/bash

# 推荐系统便携式Python环境准备脚本
# 用于下载并打包完整的Python解释器环境，使项目可在任何环境运行
# 作者：AI助手

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 显示标题
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   推荐系统便携式Python环境准备工具        ${NC}"
echo -e "${BLUE}============================================${NC}"

# 确定平台
PLATFORM=$(uname -s)
ARCH=$(uname -m)
echo -e "${GREEN}检测到平台: $PLATFORM $ARCH${NC}"

# 设置Python版本和目录
PYTHON_VERSION="3.9.12"
PORTABLE_DIR="deps/portable_python"
PYTHON_DIR="$PORTABLE_DIR/python-$PYTHON_VERSION"
DOWNLOAD_DIR="$PORTABLE_DIR/downloads"

# 创建目录结构
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$PYTHON_DIR"

# 下载并解压Python（Linux版本）
download_python_linux() {
    echo -e "${GREEN}下载Python $PYTHON_VERSION（Linux版本）...${NC}"
    PYTHON_URL="https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz"
    curl -L "$PYTHON_URL" -o "$DOWNLOAD_DIR/Python-$PYTHON_VERSION.tgz"
    
    echo -e "${GREEN}解压Python...${NC}"
    tar -xf "$DOWNLOAD_DIR/Python-$PYTHON_VERSION.tgz" -C "$DOWNLOAD_DIR"
    
    echo -e "${GREEN}配置和编译Python...${NC}"
    cd "$DOWNLOAD_DIR/Python-$PYTHON_VERSION"
    ./configure --prefix="$PWD/../../../$PYTHON_DIR" --enable-optimizations
    make -j$(nproc)
    make install
    cd "../../../"
    
    echo -e "${GREEN}便携式Python安装完成${NC}"
}

# 下载并解压Python（macOS版本）
download_python_macos() {
    echo -e "${GREEN}下载Python $PYTHON_VERSION（macOS版本）...${NC}"
    
    # 检查brew是否安装
    if ! command -v brew >/dev/null 2>&1; then
        echo -e "${RED}错误: 未找到Homebrew，请先安装${NC}"
        echo -e "${YELLOW}安装命令: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"${NC}"
        exit 1
    fi
    
    # 使用pyenv下载特定版本的Python
    if ! command -v pyenv >/dev/null 2>&1; then
        echo -e "${YELLOW}安装pyenv...${NC}"
        brew install pyenv
    fi
    
    echo -e "${GREEN}使用pyenv下载Python $PYTHON_VERSION...${NC}"
    export PYENV_ROOT="$PORTABLE_DIR/pyenv"
    mkdir -p "$PYENV_ROOT"
    eval "$(pyenv init -)"
    pyenv install "$PYTHON_VERSION"
    
    echo -e "${GREEN}复制Python安装到便携目录...${NC}"
    cp -R "$PYENV_ROOT/versions/$PYTHON_VERSION" "$PYTHON_DIR"
    
    echo -e "${GREEN}便携式Python安装完成${NC}"
}

# 创建项目激活脚本
create_activation_scripts() {
    echo -e "${GREEN}创建激活脚本...${NC}"
    
    # Linux/macOS激活脚本
    cat > activate.sh << 'EOL'
#!/bin/bash
# 推荐系统便携式Python环境激活脚本
export PORTABLE_PYTHON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/deps/portable_python"
export PATH="$PORTABLE_PYTHON_DIR/python-3.9.12/bin:$PATH"
export PYTHONHOME="$PORTABLE_PYTHON_DIR/python-3.9.12"
echo "已激活便携式Python环境"
echo "Python路径: $(which python)"
echo "Python版本: $(python --version)"
EOL
    chmod +x activate.sh
    
    # Windows激活脚本
    cat > activate.bat << 'EOL'
@echo off
REM 推荐系统便携式Python环境激活脚本
set "PORTABLE_PYTHON_DIR=%~dp0deps\portable_python"
set "PATH=%PORTABLE_PYTHON_DIR%\python-3.9.12\bin;%PATH%"
set "PYTHONHOME=%PORTABLE_PYTHON_DIR%\python-3.9.12"
echo 已激活便携式Python环境
echo Python路径: %PORTABLE_PYTHON_DIR%\python-3.9.12\bin\python.exe
python --version
EOL
}

# 更新离线部署脚本
update_deployment_script() {
    echo -e "${GREEN}更新离线部署脚本以使用便携式Python...${NC}"
    
    sed -i.bak -e 's|python -m venv|./deps/portable_python/python-'"$PYTHON_VERSION"'/bin/python -m venv|g' \
               -e 's|command -v python|command -v ./deps/portable_python/python-'"$PYTHON_VERSION"'/bin/python|g' \
               -e 's|python --version|./deps/portable_python/python-'"$PYTHON_VERSION"'/bin/python --version|g' \
               -e 's|\\$VENV_DIR/bin/python|\\$VENV_DIR/bin/python|g' offline_deploy.sh
               
    echo -e "${GREEN}部署脚本更新完成${NC}"
}

# 创建Requirements
create_requirements() {
    echo -e "${GREEN}创建或更新requirements.txt...${NC}"
    
    if [ ! -f "requirements.txt" ]; then
        cat > requirements.txt << 'EOL'
pymysql==1.1.0
numpy==1.24.3
pandas==2.0.3
schedule==1.2.0
flask==2.3.3
SQLAlchemy==2.0.23
matplotlib==3.7.2
scikit-learn==1.3.0
apscheduler==3.10.4
EOL
    else
        echo -e "${YELLOW}requirements.txt已存在，不更改${NC}"
    fi
}

# 准备离线依赖包
prepare_offline_packages() {
    echo -e "${GREEN}准备离线依赖包...${NC}"
    
    mkdir -p deps/wheels
    "$PYTHON_DIR/bin/pip" download -d deps/wheels pip setuptools wheel
    "$PYTHON_DIR/bin/pip" download -d deps/wheels -r requirements.txt
    
    echo -e "${GREEN}依赖包下载完成${NC}"
    echo -e "${GREEN}依赖包位置: deps/wheels/${NC}"
}

# 主执行逻辑
echo -e "${YELLOW}此脚本将下载并配置便携式Python环境${NC}"
echo -e "${YELLOW}继续操作? [Y/n]${NC}"
read -p "" CONTINUE
if [[ $CONTINUE == "n" || $CONTINUE == "N" ]]; then
    echo -e "${RED}操作已取消${NC}"
    exit 1
fi

# 根据平台选择下载方式
if [[ "$PLATFORM" == "Linux" ]]; then
    download_python_linux
elif [[ "$PLATFORM" == "Darwin" ]]; then
    download_python_macos
else
    echo -e "${RED}错误: 不支持的平台 $PLATFORM${NC}"
    exit 1
fi

# 创建激活脚本
create_activation_scripts

# 更新部署脚本
update_deployment_script

# 创建Requirements
create_requirements

# 准备离线包
prepare_offline_packages

echo -e "${GREEN}便携式Python环境准备完成!${NC}"
echo -e "${YELLOW}使用方法:${NC}"
echo -e "1. 在Linux/macOS上: source ./activate.sh"
echo -e "2. 在Windows上: activate.bat"
echo -e "3. 使用离线部署: ./offline_deploy.sh"
echo -e "${YELLOW}将整个项目目录（包括deps/portable_python）复制到目标服务器即可在无需安装Python的情况下运行${NC}" 