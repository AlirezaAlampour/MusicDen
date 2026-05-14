@echo off
cd /d "%~dp0"
echo Building MusicDen launcher...
pip install Pillow --quiet
python generate_icon.py
python launch_builder.py
echo.
echo Done! Find MusicDen shortcut on your Desktop.
echo Right-click it and select "Pin to taskbar"
pause
