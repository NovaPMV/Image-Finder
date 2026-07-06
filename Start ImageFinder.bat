@echo off
REM ===========================================================================
REM  Start ImageFinder (Windows launcher)
REM  ---------------------------------------------------------------------------
REM  Double-click this file to start the app. It just runs imagefinder.py with
REM  Python. Keep the black window that opens - THAT window is the program.
REM  Close it when you're done to shut the app down.
REM ===========================================================================

REM Make sure we run from the folder this .bat lives in, no matter where it's
REM launched from (so Python finds imagefinder.py sitting next to it).
cd /d "%~dp0"

echo ============================================================
echo   Starting ImageFinder...
echo   Leave this window open while you use the app.
echo   Your browser should open automatically in a moment.
echo   If it doesn't, go to http://localhost:8001 yourself.
echo ============================================================
echo.

REM Try "python" first (the usual Windows command). If that isn't found, fall
REM back to "py" (the Python launcher that some installs use instead).
python imagefinder.py
if errorlevel 1 (
    echo.
    echo "python" didn't work - trying "py" instead...
    py imagefinder.py
)

REM If we get here, the app has stopped (or failed to start). Keep the window
REM open so any error message stays readable instead of vanishing instantly.
echo.
echo ============================================================
echo   ImageFinder has stopped.
echo   If you saw an error above, check the README's
echo   "Common problems" section.
echo ============================================================
pause
