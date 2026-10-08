@echo off
setlocal EnableExtensions
set "VSENV=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VSENV%" set "VSENV=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VSENV%" exit /b 2
call "%VSENV%" >nul || exit /b 3
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
cd /d "%~dp0"
"%CUDA_PATH%\bin\nvcc.exe" -O3 -std=c++17 -arch=sm_89 --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler /Zc:preprocessor -shared -Xlinker /DEF:dsv4.def -L"%CUDA_PATH%\lib\x64" -lcublasLt -lcudart -lcuda backend_cuda_dsv4.cu -o coli_cuda_dsv4.dll
exit /b %ERRORLEVEL%
