#!/bin/bash

# 推荐系统离线部署包准备脚本
# 用于准备能在无网络环境下部署的完整包
# 作者：AI助手

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 显示标题
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}      推荐系统离线部署包准备脚本           ${NC}"
echo -e "${BLUE}============================================${NC}"

# 要清理的目录
CLEAN_DIRS=".DS_Store __pycache__ .recommend_cache .ipynb_checkpoints"

# 询问是否清理项目
echo -e "${YELLOW}是否在打包前清理项目? [y/N]${NC}"
read -p "" CLEAN
if [[ $CLEAN == "y" || $CLEAN == "Y" ]]; then
    echo -e "${GREEN}清理项目...${NC}"
    # 清理Python缓存
    find . -name __pycache__ -type d -exec rm -rf {} +
    find . -name "*.pyc" -delete
    # 清理macOS系统文件
    find . -name .DS_Store -delete
    # 清理备份文件
    find . -name "*.bak" -delete
    echo -e "${GREEN}清理完成${NC}"
fi

# 创建deps/wheels目录
WHEELS_DIR="deps/wheels"
if [ ! -d "$WHEELS_DIR" ]; then
    echo -e "${GREEN}创建${WHEELS_DIR}目录...${NC}"
    mkdir -p "$WHEELS_DIR"
fi

# 检查是否有虚拟环境
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    echo -e "${YELLOW}未找到虚拟环境，是否创建新的虚拟环境用于下载依赖? [Y/n]${NC}"
    read -p "" CREATE_VENV
    if [[ $CREATE_VENV == "n" || $CREATE_VENV == "N" ]]; then
        echo -e "${RED}中止操作${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}创建临时虚拟环境...${NC}"
    python -m venv .temp_venv
    source .temp_venv/bin/activate
    
    echo -e "${GREEN}升级pip...${NC}"
    pip install --upgrade pip setuptools wheel
    
    TEMP_VENV=true
else
    # 使用现有虚拟环境
    if [ -d "venv" ]; then
        echo -e "${GREEN}使用现有虚拟环境 'venv'...${NC}"
        source venv/bin/activate
    else
        echo -e "${GREEN}使用现有虚拟环境 '.venv'...${NC}"
        source .venv/bin/activate
    fi
    TEMP_VENV=false
fi

# 下载依赖包
echo -e "${GREEN}下载依赖包...${NC}"
pip download -d "$WHEELS_DIR" pip setuptools wheel
pip download -d "$WHEELS_DIR" -r requirements.txt

# 确认依赖包下载成功
WHEEL_COUNT=$(ls -1 "$WHEELS_DIR" | wc -l)
echo -e "${GREEN}已下载 $WHEEL_COUNT 个依赖包到 $WHEELS_DIR 目录${NC}"

# 检查部署脚本
if [ ! -f "offline_deploy.sh" ]; then
    echo -e "${RED}错误: 未找到离线部署脚本 'offline_deploy.sh'${NC}"
    exit 1
fi

# 添加执行权限
echo -e "${GREEN}添加脚本执行权限...${NC}"
chmod +x offline_deploy.sh

# 创建README文件
echo -e "${GREEN}创建离线部署说明...${NC}"
cat > OFFLINE_README.md << 'EOL'
# 推荐系统离线部署说明

本项目已配置为可在离线环境部署运行，包含所有必要的依赖包。

## 系统要求

- Python 3.9+ (必须预先安装)
- 操作系统: Linux, macOS 或 Windows

## 快速部署 (Linux/macOS)

1. 解压部署包
2. 进入项目目录
3. 运行部署脚本:

```bash
chmod +x offline_deploy.sh  # 确保有执行权限
./offline_deploy.sh
```

## 快速部署 (Windows)

1. 解压部署包
2. 进入项目目录
3. 双击运行 `offline_deploy_windows.bat`

## 常用参数

可以在部署脚本后添加以下参数:

- `--init-db`: 初始化数据库 (首次部署必须使用)
- `--debug`: 以调试模式运行
- `--no-scheduler`: 禁用调度任务
- `--no-log`: 不保存日志文件

例如:
```bash
./offline_deploy.sh --init-db --debug
```

## 注意事项

1. 项目将自动创建虚拟环境并安装所有依赖
2. 依赖包储存在 `deps/wheels` 目录中，请勿删除
3. 日志文件保存在 `logs` 目录中

## 故障排除

如果遇到依赖安装问题，请确保:

1. Python版本为3.9或更高
2. deps/wheels目录完整无损
3. 对虚拟环境目录有写入权限

## 联系方式

如有问题，请联系技术支持。
EOL

# 如果创建了临时虚拟环境，则退出并删除
if [ "$TEMP_VENV" = true ]; then
    echo -e "${GREEN}清理临时虚拟环境...${NC}"
    deactivate
    rm -rf .temp_venv
else
    deactivate
fi

echo -e "${GREEN}准备完成!${NC}"
echo -e "${YELLOW}您现在可以将整个项目目录打包部署到离线环境.${NC}"
echo -e "${YELLOW}在目标环境中运行:${NC}"
echo -e "  Linux/macOS: ./offline_deploy.sh"
echo -e "  Windows:     双击 offline_deploy_windows.bat" 