@echo off
REM Run this from the project folder. Creates a venv, installs deps, and launches Streamlit.
cd /d "%~dp0"

















streamlit run app.py
:: Run Streamlit app)    set REPLICATE_API_TOKEN=%REPLICATE_API_TOKEN%if defined REPLICATE_API_TOKEN (set /p REPLICATE_API_TOKEN=Enter Replicate API token (or press Enter to skip): 
:: Optionally set Replicate API token for this run (press Enter to skip and use Streamlit secrets)python -m pip install -r requirements.txtpython -m pip install --upgrade pip
:: Upgrade pip and install requirementscall .venv\Scripts\activate.bat
:: Activate venv)    python -m venv .venvif not exist .venv (:: Create venv if missing