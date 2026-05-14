@echo off
echo Checking for known vulnerabilities...
pip install pip-audit --quiet
pip-audit -r requirements.txt
pause
