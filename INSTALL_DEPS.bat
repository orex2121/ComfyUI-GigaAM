@echo off
chcp 65001 > nul
setlocal

set "PYTHON_EXE=%~dp0..\..\..\python_embeded\python.exe"

if not exist "%PYTHON_EXE%" (
    echo ERROR: Python not found at "%PYTHON_EXE%"
    goto :error
)

echo Installing GigaAM without dependencies...
"%PYTHON_EXE%" -m pip install "gigaam[longform] @ git+https://github.com/salute-developers/GigaAM.git" --no-deps
if errorlevel 1 goto :error

echo.
echo Installing hydra-core...
"%PYTHON_EXE%" -m pip install hydra-core
if errorlevel 1 goto :error

echo.
echo Installing pyannote.audio...
"%PYTHON_EXE%" -m pip install pyannote.audio
if errorlevel 1 goto :error

echo.
echo Checking GigaAM import...
"%PYTHON_EXE%" -c "import gigaam; print('GigaAM OK:', gigaam.__file__)"
if errorlevel 1 goto :error

echo.
echo Installation complete. Restart ComfyUI.
pause
exit /b 0

:error
echo.
echo Installation or import check failed. See the error above.
pause
exit /b 1