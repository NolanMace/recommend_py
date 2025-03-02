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
2. 依赖包储存在 `wheels` 目录中，请勿删除
3. 日志文件保存在 `logs` 目录中

## 故障排除

如果遇到依赖安装问题，请确保:

1. Python版本为3.9或更高
2. wheels目录完整无损
3. 对虚拟环境目录有写入权限

## 联系方式

如有问题，请联系技术支持。
