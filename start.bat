@echo off
cd /d "%~dp0"
for /f %%i in ('python -c "import json,os; p='config.json'; d={'port':7337,'default_download_path':'D:/Music/DeezerDownloads','default_quality':'FLAC','library_folders':[]}; json.dump(d,open(p,'w')) if not os.path.exists(p) else None; print(json.load(open(p)).get('port',7337))"') do set PORT=%%i
start "" uvicorn main:app --host 127.0.0.1 --port %PORT%
timeout /t 2 /nobreak >nul
start http://127.0.0.1:%PORT%
