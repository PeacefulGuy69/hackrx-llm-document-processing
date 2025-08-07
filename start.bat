@echo off
echo Starting LLM Document Processing System...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Check if .env file exists
if not exist ".env" (
    echo Creating .env file from template...
    copy .env.example .env
    echo.
    echo ⚠️  IMPORTANT: Please edit .env file and add your OpenAI API key!
    echo    Open .env and set OPENAI_API_KEY=your_api_key_here
    echo.
    pause
)

REM Start the server
echo Starting FastAPI server...
python main.py
