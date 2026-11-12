@echo off
setlocal EnableDelayedExpansion
rem ==============================================================================
rem OceanGo V2 — Automated 3-Stage Curriculum Runner for Windows 11 (CMD / PowerShell)
rem Stages: Stage 1 (9x9) -> Stage 2 (13x13) -> Stage 3 (19x19)
rem
rem Usage:
rem   train_curriculum.bat                     (Flagship 18x256 preset + --compile)
rem   train_curriculum.bat --preset balanced   (Balanced 15x192 preset for 8-12GB GPUs)
rem   train_curriculum.bat --no-compile        (Force eager execution)
rem ==============================================================================

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python was not found in PATH. Run setup_venv.bat first.
    exit /b 1
)

set "SAVE_DIR=checkpoints"
set "COMPILE_FLAG=--compile"
set "PRESET=flagship"
set "EXTRA_ARGS="

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--no-compile" (
    set "COMPILE_FLAG="
    shift
    goto parse_args
)
if /i "%~1"=="--compile" (
    set "COMPILE_FLAG=--compile"
    shift
    goto parse_args
)
if /i "%~1"=="--save-dir" (
    set "SAVE_DIR=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--preset" (
    set "PRESET=%~2"
    shift
    shift
    goto parse_args
)
set "EXTRA_ARGS=!EXTRA_ARGS! %~1"
shift
goto parse_args
:args_done

if /i "%PRESET%"=="balanced" (
    set "S1_BLOCKS=15"
    set "S1_CHANNELS=192"
    set "S1_GAMES=500"
    set "S1_SIMS=600"
    set "S1_ITERS=30"
    set "S2_GAMES=400"
    set "S2_SIMS=600"
    set "S2_ITERS=35"
    set "S3_GAMES=250"
    set "S3_SIMS=800"
    set "S3_ITERS=50"
) else (
    set "S1_BLOCKS=18"
    set "S1_CHANNELS=256"
    set "S1_GAMES=600"
    set "S1_SIMS=800"
    set "S1_ITERS=40"
    set "S2_GAMES=500"
    set "S2_SIMS=800"
    set "S2_ITERS=45"
    set "S3_GAMES=300"
    set "S3_SIMS=1000"
    set "S3_ITERS=60"
)

echo ======================================================================
echo OceanGo V2 Automated 3-Stage Curriculum ^(Windows 11^)
echo Preset      : %PRESET% ^(%S1_BLOCKS% blocks x %S1_CHANNELS% channels^)
echo Save Dir    : %SAVE_DIR%
if defined COMPILE_FLAG (echo Compile     : %COMPILE_FLAG%) else (echo Compile     : disabled)
echo Started At  : %DATE% %TIME%
echo ======================================================================

rem ------------------------------------------------------------------------------
rem Stage 1: 9x9 — Tactical Reading, Captures, Ko & Life-and-Death
rem ------------------------------------------------------------------------------
call :get_ckpt_state
set "RUN_S1=0"
if !CKPT_BOARD! equ 0 set "RUN_S1=1"
if !CKPT_BOARD! equ 9 if !CKPT_STAGE_ITER! lss %S1_ITERS% set "RUN_S1=1"

if "!RUN_S1!"=="1" (
    echo.
    echo ^>^>^> [Stage 1/3] Launching 9x9 Curriculum Stage ^(Target: %S1_ITERS% stage iters^)...
    python -u OceanGo.py --train --save-dir "%SAVE_DIR%" --board-size 9 --blocks %S1_BLOCKS% --channels %S1_CHANNELS% --games %S1_GAMES% --sims %S1_SIMS% --iters %S1_ITERS% --lr 0.01 --replay-ratio 1.5 --eval-interval 5 --eval-games 20 %COMPILE_FLAG% %EXTRA_ARGS%
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Stage 1 exited with code !ERRORLEVEL!. Stopping curriculum pipeline.
        exit /b !ERRORLEVEL!
    )
) else (
    echo.
    echo ^>^>^> [Stage 1/3] Skipping 9x9 ^(already at !CKPT_BOARD!x!CKPT_BOARD!, iter=!CKPT_ITER!, stage_iter=!CKPT_STAGE_ITER!^).
)

rem ------------------------------------------------------------------------------
rem Stage 2: 13x13 — Joseki, Ladders & Whole-Board Influence
rem ------------------------------------------------------------------------------
call :get_ckpt_state
set "RUN_S2=0"
if !CKPT_BOARD! equ 9 set "RUN_S2=1"
if !CKPT_BOARD! equ 13 if !CKPT_STAGE_ITER! lss %S2_ITERS% set "RUN_S2=1"

if "!RUN_S2!"=="1" (
    echo.
    echo ^>^>^> [Stage 2/3] Launching 13x13 Curriculum Stage ^(Target: %S2_ITERS% stage iters^)...
    python -u OceanGo.py --train --save-dir "%SAVE_DIR%" --board-size 13 --games %S2_GAMES% --sims %S2_SIMS% --iters %S2_ITERS% --lr 0.01 --replay-ratio 1.5 --eval-interval 5 --eval-games 20 %COMPILE_FLAG% %EXTRA_ARGS%
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Stage 2 exited with code !ERRORLEVEL!. Stopping curriculum pipeline.
        exit /b !ERRORLEVEL!
    )
) else (
    echo.
    echo ^>^>^> [Stage 2/3] Skipping 13x13 ^(already at !CKPT_BOARD!x!CKPT_BOARD!, iter=!CKPT_ITER!, stage_iter=!CKPT_STAGE_ITER!^).
)

rem ------------------------------------------------------------------------------
rem Stage 3: 19x19 — Full-Board Fuseki, Fighting & Endgame Yose
rem ------------------------------------------------------------------------------
call :get_ckpt_state
set "RUN_S3=0"
if !CKPT_BOARD! equ 13 set "RUN_S3=1"
if !CKPT_BOARD! equ 19 if !CKPT_STAGE_ITER! lss %S3_ITERS% set "RUN_S3=1"

if "!RUN_S3!"=="1" (
    echo.
    echo ^>^>^> [Stage 3/3] Launching 19x19 Curriculum Stage ^(Target: %S3_ITERS% stage iters^)...
    python -u OceanGo.py --train --save-dir "%SAVE_DIR%" --board-size 19 --games %S3_GAMES% --sims %S3_SIMS% --iters %S3_ITERS% --lr 0.005 --replay-ratio 1.8 --eval-interval 5 --eval-games 20 %COMPILE_FLAG% %EXTRA_ARGS%
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Stage 3 exited with code !ERRORLEVEL!. Stopping curriculum pipeline.
        exit /b !ERRORLEVEL!
    )
) else (
    echo.
    echo ^>^>^> [Stage 3/3] Skipping 19x19 ^(already completed: !CKPT_BOARD!x!CKPT_BOARD!, iter=!CKPT_ITER!, stage_iter=!CKPT_STAGE_ITER!^).
)

echo.
echo ======================================================================
echo All 3 Curriculum Stages Complete at %DATE% %TIME%!
echo Inspect checkpoints : python policy_print.py --save-dir "%SAVE_DIR%"
echo Play vs 19x19 AI    : python OceanGo.py --play --sims 1200 --play-symmetries 8
echo ======================================================================
exit /b 0

:get_ckpt_state
set "CKPT_BOARD=0"
set "CKPT_ITER=0"
set "CKPT_STAGE_ITER=0"
for /f "tokens=1,2,3,4" %%A in ('python -c "import contextlib,os,sys; f=contextlib.redirect_stdout(sys.stderr); f.__enter__(); from OceanGo import resolve_latest_v2_checkpoint,load_checkpoint_auto,is_state_dict_finite; p=resolve_latest_v2_checkpoint(r'%SAVE_DIR%'); m=load_checkpoint_auto(p) if (p and os.path.isfile(p)) else {}; ok=(m.get('version')==2 and is_state_dict_finite(m.get('state_dict'))); b=int(m.get('board_size',0)) if ok else 0; it=int(m.get('iteration',0)) if ok else 0; st=int(m.get('stage_iter',it)) if ok else 0; f.__exit__(None,None,None); print('CKPT_STATE',b,it,st)"') do (
    if "%%A"=="CKPT_STATE" (
        set "CKPT_BOARD=%%B"
        set "CKPT_ITER=%%C"
        set "CKPT_STAGE_ITER=%%D"
    )
)
exit /b 0
