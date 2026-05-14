import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
START_BAT = os.path.join(APP_DIR, "start.bat")

# ── VBScript: runs start.bat with no visible console window ──────────────────
vbs = os.path.join(APP_DIR, "MusicDen_launcher.vbs")
with open(vbs, "w") as f:
    f.write(
        f'Set WshShell = CreateObject("WScript.Shell")\n'
        f'WshShell.CurrentDirectory = "{APP_DIR}"\n'
        f'WshShell.Run "cmd /c start.bat", 0, False\n'
    )
print(f"Created: {vbs}")

# ── PowerShell: creates a pinnable .lnk shortcut on the Desktop ──────────────
icon_path = os.path.join(APP_DIR, "icon.ico")

ps1 = os.path.join(APP_DIR, "create_shortcut.ps1")
with open(ps1, "w", encoding="utf-8") as f:
    f.write(
        f'$desktop = [Environment]::GetFolderPath("Desktop")\n'
        f'$shortcut_path = Join-Path $desktop "MusicDen.lnk"\n'
        f'$ws = New-Object -ComObject WScript.Shell\n'
        f'$s = $ws.CreateShortcut($shortcut_path)\n'
        f'$s.TargetPath = "wscript.exe"\n'
        f'$s.Arguments = \'"{vbs}"\'\n'
        f'$s.WorkingDirectory = "{APP_DIR}"\n'
        f'$s.Description = "Launch MusicDen"\n'
        f'if (Test-Path "{icon_path}") {{ $s.IconLocation = "{icon_path}" }}\n'
        f'$s.Save()\n'
        f'Write-Host "Shortcut created at: $shortcut_path"\n'
        f'Write-Host "Right-click it on the Desktop and choose \'Pin to taskbar\'"\n'
    )
print(f"Created: {ps1}")

# ── Run the PowerShell script ─────────────────────────────────────────────────
ret = os.system(f'powershell -ExecutionPolicy Bypass -File "{ps1}"')
if ret != 0:
    print("Warning: PowerShell script exited with code", ret)

print("\nDone. Find MusicDen on your Desktop — right-click → Pin to taskbar.")
