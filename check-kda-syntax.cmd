@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cd /d C:\src\colibri-kda-batch-20261008\c
cl /nologo /std:c11 /Zs /W3 glm53.c
exit /b %ERRORLEVEL%
