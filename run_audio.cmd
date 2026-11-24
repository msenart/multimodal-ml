@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project Python environment is missing. See docs/audio/README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" "scripts\run_audio.py" %*
set "audio_exit=%errorlevel%"
if "%~1"=="" pause
exit /b %audio_exit%
