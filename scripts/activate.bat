@echo off
REM Virtual Environment Activation Script for Windows
REM This script activates the pyenv-virtualenv environment for the AI News Article Query System

echo AI News Article Query System - Environment Activation
echo ========================================================

REM Environment name
set ENV_NAME=ai-news-query

REM Check if pyenv is installed
where pyenv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: pyenv is not installed or not in PATH
    echo Please install pyenv-win: https://github.com/pyenv-win/pyenv-win
    exit /b 1
)

REM Check if the virtual environment exists
pyenv virtualenvs | findstr /C:"%ENV_NAME%" >nul
if %ERRORLEVEL% NEQ 0 (
    echo Virtual environment '%ENV_NAME%' not found.
    echo Please create it first using:
    echo   pyenv virtualenv 3.10.15 %ENV_NAME%
    echo   pyenv local %ENV_NAME%
    exit /b 1
)

REM Activate the environment
echo Activating virtual environment: %ENV_NAME%
pyenv activate %ENV_NAME%

if %ERRORLEVEL% EQU 0 (
    echo Environment activated successfully!
    echo.
    python --version
    where python
    echo.
    echo To deactivate, run: pyenv deactivate
) else (
    echo Failed to activate environment
    exit /b 1
)
