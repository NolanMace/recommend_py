@echo off
REM 推荐系统便携式Python环境激活脚本
set "PORTABLE_PYTHON_DIR=%~dp0deps\portable_python"
set "PATH=%PORTABLE_PYTHON_DIR%\python-3.9.12\bin;%PATH%"
set "PYTHONHOME=%PORTABLE_PYTHON_DIR%\python-3.9.12"
echo 已激活便携式Python环境
echo Python路径: %PORTABLE_PYTHON_DIR%\python-3.9.12\bin\python.exe
python --version
