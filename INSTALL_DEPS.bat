@echo off
chcp 65001 > nul
set PYTHON_EXE=..\..\..\python_embeded\python.exe

echo Installing GigaAM without changing current dependencies...
%PYTHON_EXE% -m pip install "gigaam[longform] @ git+https://github.com/salute-developers/GigaAM.git" --no-deps

echo.
echo Installing pyannote.audio...
%PYTHON_EXE% -m pip install pyannote.audio

echo.
echo Installation complete.
pause