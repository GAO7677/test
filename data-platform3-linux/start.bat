@echo off
chcp 65001 >nul
title 数据中台一键启动
cd /d %~dp0backend

echo ============================================
echo   航空数据中台  一键启动
echo   后端启动后将自动托管前端页面
echo ============================================
echo.

REM 配置文件不存在时自动从模板复制（首次运行）
if not exist config.json (
    echo [提示] 未找到 config.json，已从模板复制。
    copy /y config.json.example config.json >nul
    echo        如需修改数据库连接，请编辑 backend\config.json 后重新运行本脚本。
    echo.
)

REM 优先使用 py 启动器，其次 python
where py >nul 2>nul
if %errorlevel%==0 (
    py -m app.main
) else (
    python -m app.main
)

echo.
echo [提示] 服务已退出。如为异常退出，请把上方报错信息反馈给管理员。
pause
