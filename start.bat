@echo off
chcp 65001 >nul
setlocal EnableExtensions

cd /d "%~dp0"

if not defined FLASK_PORT set "FLASK_PORT=5001"
set "APP_ROOT=%~dp0"
set "BACKEND_DIR=%APP_ROOT%backend"
set "HEALTH_URL=http://127.0.0.1:%FLASK_PORT%/api/health"
set "DEFAULT_CONDA_ENV=ml0"
set "PY_CMD=python"
set "ENV_DESC=System Python"

echo.
echo ===== Indeterminate - Flask + Streamlit =====
echo.

call :try_default_conda
if "%ENV_READY%"=="1" goto start

echo   Select Python environment:
echo     [1] System Python
echo     [2] Conda / Miniconda
echo     [3] venv / virtualenv
echo.
set /p choice="Enter number (1/2/3): "

if "%choice%"=="2" goto conda
if "%choice%"=="3" goto venv
goto system_python

:try_default_conda
where conda >nul 2>&1
if errorlevel 1 goto default_conda_direct

echo Checking default conda environment: %DEFAULT_CONDA_ENV% ...
conda run -n %DEFAULT_CONDA_ENV% python -c "import sys" >nul 2>&1
if errorlevel 1 goto default_conda_direct

set "ENV_READY=1"
set "ENV_DESC=Conda: %DEFAULT_CONDA_ENV%"
set "PY_CMD=conda run -n %DEFAULT_CONDA_ENV% python"
echo [OK] Using %ENV_DESC%
echo.
exit /b 0

:default_conda_direct
call :find_conda_env_python "%DEFAULT_CONDA_ENV%"
if not defined FOUND_PY (
    echo [INFO] Conda environment "%DEFAULT_CONDA_ENV%" was not found or is not usable.
    echo.
    exit /b 0
)

set "ENV_READY=1"
set "ENV_DESC=Conda direct: %DEFAULT_CONDA_ENV%"
set "PY_CMD=""%FOUND_PY%"""
echo [OK] Using %ENV_DESC%
echo.
exit /b 0

:system_python
set "ENV_READY=1"
set "ENV_DESC=System Python"
set "PY_CMD=python"
goto start

:conda
echo.
set /p env_name="Conda environment name (default: ml0): "
if "%env_name%"=="" set "env_name=ml0"

where conda >nul 2>&1
if errorlevel 1 goto conda_direct

conda run -n %env_name% python -c "import sys" >nul 2>&1
if errorlevel 1 goto conda_direct

set "ENV_READY=1"
set "ENV_DESC=Conda: %env_name%"
set "PY_CMD=conda run -n %env_name% python"
goto start

:conda_direct
call :find_conda_env_python "%env_name%"
if not defined FOUND_PY (
    echo [FAIL] Cannot run Python in conda environment: %env_name%
    echo        Check: conda info --envs
    pause
    exit /b 1
)

set "ENV_READY=1"
set "ENV_DESC=Conda direct: %env_name%"
set "PY_CMD=""%FOUND_PY%"""
goto start

:venv
echo.
set /p venv_path="venv folder path (e.g. venv): "
if exist "%venv_path%\Scripts\python.exe" (
    set "ENV_READY=1"
    set "ENV_DESC=venv: %venv_path%"
    set "PY_CMD=""%venv_path%\Scripts\python.exe"""
    goto start
)

echo [FAIL] Cannot find %venv_path%\Scripts\python.exe
pause
exit /b 1

:find_conda_env_python
set "FOUND_PY="
for %%p in (
    "%USERPROFILE%\.conda\envs\%~1\python.exe"
    "%USERPROFILE%\miniconda3\envs\%~1\python.exe"
    "%USERPROFILE%\anaconda3\envs\%~1\python.exe"
) do (
    if exist "%%~p" (
        set "FOUND_PY=%%~p"
        exit /b 0
    )
)
exit /b 1

:start
echo.
echo Environment     : %ENV_DESC%
echo Flask backend  : http://localhost:%FLASK_PORT%
echo Streamlit      : http://localhost:8501
echo.

echo [1/3] Checking Python modules...
%PY_CMD% -c "import flask, flask_cors, streamlit, pandas, sklearn, plotly, requests, torch; print('dependencies ok')"
if errorlevel 1 (
    echo.
    echo [FAIL] Required modules are missing in %ENV_DESC%.
    echo        Run setup.bat or install requirements in this environment.
    pause
    exit /b 1
)

echo.
echo [2/3] Starting Flask backend...
start "Indeterminate Flask" cmd /k "cd /d ""%BACKEND_DIR%"" && set ""FLASK_PORT=%FLASK_PORT%"" && %PY_CMD% app.py"

echo Waiting for backend health check...
call :wait_for_backend
if errorlevel 1 (
    echo.
    echo [FAIL] Flask backend did not become healthy: %HEALTH_URL%
    echo        Check the Flask window for the Python traceback.
    pause
    exit /b 1
)

echo [OK] Backend is healthy.
echo.
echo [3/3] Starting Streamlit frontend...
set "INDETERMINATE_API_BASE=http://127.0.0.1:%FLASK_PORT%/api"
%PY_CMD% -m streamlit run main.py

pause
exit /b 0

:wait_for_backend
for /l %%i in (1,1,30) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-RestMethod -Uri '%HEALTH_URL%' -TimeoutSec 2; if ($r.status -eq 'ok') { exit 0 } } catch { }; exit 1" >nul 2>&1
    if not errorlevel 1 exit /b 0
    timeout /t 1 /nobreak >nul
)
exit /b 1
