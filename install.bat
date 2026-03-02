@echo off
echo Installing Agentic Market Risk Forecaster dependencies...
echo.

echo Step 1: Installing core packages...
pip install -r requirements.txt

echo.
echo Step 2: Upgrading openai and litellm to versions compatible with Groq routing...
echo (crewai pins openai 1.x but works at runtime with 2.x -- this fixes the conflict)
pip install "openai>=2.8.0" litellm --upgrade

echo.
echo Installation complete.
echo.
echo To run the dashboard:
echo   python -m streamlit run app/main.py
echo.
pause
