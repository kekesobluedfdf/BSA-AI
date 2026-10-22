@echo off
cd /d "%~dp0.."
"%~dp0.venv\Scripts\python.exe" -m streamlit run "%~dp0ui\app.py" --server.port 8501
pause
