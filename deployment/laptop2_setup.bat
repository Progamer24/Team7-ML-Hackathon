@echo off
REM ============================================================================
REM PathSnap - Laptop 2 (Evaluator UI Deployment)
REM Hosts the Streamlit evaluator interface communicating with Laptop 1 API.
REM ============================================================================

title PathSnap - Laptop 2 (Evaluator UI)
echo ============================================================================
echo   PATHSNAP: LAPTOP 2 DEPLOYMENT (STREAMLIT EVALUATOR UI)
echo ============================================================================
echo.

REM 1. Activate Virtual Environment if available
if exist .venv\Scripts\activate.bat (
    echo Activating Python virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else (
    echo Using system Python installation...
)

REM 2. Instructions for API URL
echo [INSTRUCTION] In the Streamlit sidebar, enter the Laptop 1 API address:
echo Example: http://192.168.1.100:8000
echo.

REM 3. Launch Streamlit UI
echo Starting Streamlit on http://localhost:8501...
streamlit run ui\app.py --server.port 8501 --server.address 0.0.0.0

pause

