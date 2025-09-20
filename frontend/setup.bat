@echo off
echo.
echo ================================
echo  BDD Agentic Workflow - Angular
echo ================================
echo.
echo Installing dependencies...
echo.

call npm install

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install dependencies!
    echo Please make sure you have Node.js and npm installed.
    echo.
    pause
    exit /b 1
)

echo.
echo ================================
echo  Installation Complete!
echo ================================
echo.
echo To start the development server, run:
echo   npm start
echo.
echo Or manually run:
echo   ng serve
echo.
echo The application will be available at:
echo   http://localhost:4200
echo.
pause