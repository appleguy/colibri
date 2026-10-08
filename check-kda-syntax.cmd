@echo off
setlocal
set "CC=C:\tools\w64devkit-2.10.0\w64devkit\bin\gcc.exe"
set "ROOT=C:\src\colibri-kda-batch-20261008\c"
"%CC%" -std=c11 -D_FILE_OFFSET_BITS=64 -fsyntax-only -I "%ROOT%" "%ROOT%\glm53.c"
exit /b %ERRORLEVEL%
