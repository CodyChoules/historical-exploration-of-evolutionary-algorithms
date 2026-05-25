@echo off
setlocal
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\..\C964_Computer_Science_Capstone
"%PROJECT_ROOT%\manim.venv\Scripts\python.exe" "%SCRIPT_DIR%clean_experiment_data.py"
endlocal
