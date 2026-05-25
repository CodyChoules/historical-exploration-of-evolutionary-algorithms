@echo off
setlocal
set PROJECT_ROOT=%~dp0..\..\C964_Computer_Science_Capstone
pushd "%PROJECT_ROOT%"
"%PROJECT_ROOT%\manim.venv\Scripts\python.exe" optimizationlab\vizualize_experiment.py --experiment md2 --seed 229
popd
endlocal
