@echo off
:: 推荐系统启动脚本 (Windows)
:: 作者: AI助手

title 推荐系统

:: 设置颜色
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "NC=[0m"

:: 显示标题
echo %BLUE%============================================%NC%
echo %BLUE%         推荐系统启动脚本 (Windows)          %NC%
echo %BLUE%============================================%NC%

:: 检测虚拟环境
if not exist ".venv" (
    echo %RED%错误: 未找到虚拟环境 (.venv 目录)%NC%
    echo %YELLOW%请先创建虚拟环境：%NC%
    echo python -m venv .venv
    echo .venv\Scripts\activate
    echo pip install -r requirements.txt
    pause
    exit /b 1
)

:: 设置日志文件
set "LOG_DIR=logs"
set "TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "LOG_FILE=%LOG_DIR%\recommend_system_%TIMESTAMP%.log"

:: 创建日志目录
if not exist "%LOG_DIR%" (
    echo %YELLOW%创建日志目录: %LOG_DIR%%NC%
    mkdir "%LOG_DIR%"
)

:: 激活虚拟环境
echo %GREEN%正在激活虚拟环境...%NC%
call .venv\Scripts\activate.bat

:: 检查虚拟环境是否激活成功
if %ERRORLEVEL% neq 0 (
    echo %RED%错误: 无法激活虚拟环境%NC%
    pause
    exit /b 1
)

:: 显示Python版本
echo %GREEN%使用Python版本:%NC%
python --version

:: 解析命令行参数
set "INIT_DB="
set "DEBUG="
set "NO_SCHEDULER="
set "NO_LOG="
set "HELP="

:parse_args
if "%~1"=="" goto execute
if "%~1"=="--init-db" set "INIT_DB=1"
if "%~1"=="--debug" set "DEBUG=1"
if "%~1"=="--no-scheduler" set "NO_SCHEDULER=1"
if "%~1"=="--no-log" set "NO_LOG=1"
if "%~1"=="--help" set "HELP=1"
shift
goto parse_args

:: 显示帮助信息
:show_help
echo %YELLOW%使用方法:%NC%
echo   run.bat [选项]
echo.
echo %YELLOW%可用选项:%NC%
echo   --help          显示此帮助信息
echo   --init-db       初始化数据库表结构
echo   --debug         启用调试模式
echo   --no-scheduler  禁用调度任务服务
echo   --no-log        不保存日志文件
echo.
echo %YELLOW%示例:%NC%
echo   run.bat --init-db      # 初始化数据库并启动系统
echo   run.bat --debug        # 以调试模式启动系统
goto :eof

:: 执行命令
:execute
if defined HELP (
    call :show_help
    goto end
)

:: 准备启动参数
set "RUN_ARGS="
if defined INIT_DB set "RUN_ARGS=%RUN_ARGS% --init-db"
if defined DEBUG set "RUN_ARGS=%RUN_ARGS% --debug"
if defined NO_SCHEDULER set "RUN_ARGS=%RUN_ARGS% --no-scheduler"

:: 判断是否记录日志
if defined NO_LOG (
    echo %GREEN%启动推荐系统...%NC%
    echo %YELLOW%参数: %RUN_ARGS%%NC%
    python main.py %RUN_ARGS%
) else (
    echo %GREEN%启动推荐系统...%NC%
    echo %YELLOW%参数: %RUN_ARGS% --log-file %LOG_FILE%%NC%
    echo %GREEN%日志将保存到: %LOG_FILE%%NC%
    python main.py %RUN_ARGS% --log-file "%LOG_FILE%"
    
    :: 系统退出后显示日志路径
    echo %GREEN%系统已退出，查看完整日志:%NC%
    echo type "%LOG_FILE%"
)

:end
:: 退出虚拟环境
call deactivate
echo.
echo %GREEN%按任意键退出...%NC%
pause > nul 