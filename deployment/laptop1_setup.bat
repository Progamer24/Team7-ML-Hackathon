@echo off
REM ============================================================================
REM PathSnap - Laptop 1 (Model and API Server Deployment)
REM Hosts the trained classifier, HMM/Viterbi engine, and FastAPI endpoints.
REM ============================================================================

title PathSnap - Laptop 1 (API Server)
echo ============================================================================
echo   PATHSNAP: LAPTOP 1 DEPLOYMENT (FASTAPI MODEL SERVER)
echo ============================================================================
echo.

REM 1. Display Local IP for Laptop 2 configuration
echo Finding local IPv4 address for Laptop 2 connection...
for /f "tokens=4" %%a in ('route print^|findstr 0.0.0.0.*0.0.0.0') do (
    echo [INFO] Detected Primary Interface Gateway
)
ipconfig | findstr /i "IPv4"
echo.
echo NOTE: Provide one of the above IPv4 addresses to the Laptop 2 UI operator.
echo Ensure Windows Firewall allows incoming connections on port 8000.
echo.

REM 2. Activate Virtual Environment if available
if exist .venv\Scripts\activate.bat (
    echo Activating Python virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else (
    echo Using system Python installation...
)

REM 3. Verify Model Artifact
if not exist artifacts\model.joblib (
    echo [WARNING] artifacts\model.joblib not found. Generating baseline artifacts...
    python -m src.training.pipeline
)

REM 4. Launch FastAPI Server
echo Starting Uvicorn server on http://0.0.0.0:8000...
echo Keep this command prompt open to show evaluators real-time API logs.
echo.
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

pause

