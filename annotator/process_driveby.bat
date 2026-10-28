@echo off
REM ============================================================
REM  DriveBy — one-click pipeline
REM  Remux .h264 -> .mp4, detect, render. Skips finished work,
REM  so it is safe to stop and re-run any time.
REM ============================================================

setlocal enabledelayedexpansion

REM ---------- SETTINGS ----------
set "FOLDER="
if not "%~1"=="" set "FOLDER=%~1"
if not defined FOLDER set /p "FOLDER=Drag a project folder here and press Enter: "
set "FOLDER=%FOLDER:"=%"
if not exist "%FOLDER%\" (echo Folder not found: %FOLDER% & pause & exit /b 1)
set "PY=%USERPROFILE%\driveby-env\Scripts\python.exe"
set "SCRIPTS=%~dp0"
set "FPS=30"
set "DET_CONF=0.05"
set "DET_IMGSZ=960"
set "MODEL=yolo26l.pt"
set "REN_CONF=0.4"
set "REN_MINFRAMES=30"
REM ------------------------------

echo.
echo ===========================================
echo   DriveBy pipeline
echo   folder: %FOLDER%
echo   started: %DATE% %TIME%
echo ===========================================

cd /d "%FOLDER%" || (echo Could not open %FOLDER% & pause & exit /b 1)

REM ---------------------------------------------------------
REM  STEP 1 - remux any .h264 that has no .mp4 yet
REM ---------------------------------------------------------
echo.
echo --- STEP 1: remux ---
set /a NREMUX=0
for %%F in (*.h264) do (
    if not exist "%%~nF.mp4" (
        echo   remuxing %%~nxF
        ffmpeg -hide_banner -loglevel error -framerate %FPS% -i "%%F" -c copy "%%~nF.mp4"
        if errorlevel 1 (echo   [FAILED] %%~nxF) else (set /a NREMUX+=1)
    )
)
echo   %NREMUX% file^(s^) remuxed

REM ---------------------------------------------------------
REM  STEP 2 - detect. detect.py takes the whole folder and
REM  skips anything that already has a sidecar.
REM ---------------------------------------------------------
echo.
echo --- STEP 2: detect ---
"%PY%" "%SCRIPTS%\detect.py" "%FOLDER%" --model %MODEL% --conf %DET_CONF% --imgsz %DET_IMGSZ%

REM ---------------------------------------------------------
REM  STEP 3 - render anything with a sidecar but no annotated
REM  output. Skips .annotated.mp4 files themselves.
REM ---------------------------------------------------------
echo.
echo --- STEP 3: render ---
set /a NREN=0
for %%F in (*.mp4) do (
    echo %%~nF | find /i ".annotated" >nul
    if errorlevel 1 (
        if exist "%%~nF.bytetrack.jsonl" (
            if not exist "%%~nF.annotated.mp4" (
                echo   rendering %%~nxF
                "%PY%" "%SCRIPTS%\render.py" "%%F" --conf %REN_CONF% --min-frames %REN_MINFRAMES%
                set /a NREN+=1
            )
        )
    )
)
echo   %NREN% file^(s^) rendered

echo.
echo ===========================================
echo   DONE  %DATE% %TIME%
echo ===========================================
echo.
pause
