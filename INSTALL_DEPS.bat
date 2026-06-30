@echo off
chcp 65001 > nul
set PYTHON_EXE=..\..\..\python_embeded\python.exe

echo Установка GigaAM без изменения текущих зависимостей...
%PYTHON_EXE% -m pip install "gigaam[longform] @ git+https://github.com/salute-developers/GigaAM.git" --no-deps

echo.
echo Установка pyannote.audio...
%PYTHON_EXE% -m pip install pyannote.audio

echo.
echo Установка завершена.
pause