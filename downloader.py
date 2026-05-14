import asyncio
import json
import os
import subprocess
import sys
import threading
import uuid
from typing import Dict, Optional

_jobs: Dict[str, dict] = {}
_DEEMIX_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".deemix_config")


def start_download(url: str, quality: str, output_path: str, arl_token: str = "") -> str:
    """Start a deemix download subprocess. Callers must pre-validate all arguments."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"lines": [], "done": False, "exit_code": None, "process": None}

    def run():
        # Write ARL to deemix config dir so deemix can authenticate
        os.makedirs(_DEEMIX_CONFIG_DIR, exist_ok=True)
        with open(os.path.join(_DEEMIX_CONFIG_DIR, "config.json"), "w") as cf:
            json.dump({"arl": arl_token}, cf)

        cmd = [
            sys.executable, "-m", "deemix", "--portable",
            "--config-dir", _DEEMIX_CONFIG_DIR,
            "-b", quality, "-p", output_path, url,
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            _jobs[job_id]["process"] = proc
            for line in proc.stdout:
                _jobs[job_id]["lines"].append(line.rstrip())
            proc.wait()
            _jobs[job_id]["exit_code"] = proc.returncode
        except Exception as e:
            _jobs[job_id]["lines"].append(f"[ERROR] {e}")
            _jobs[job_id]["exit_code"] = -1
        finally:
            _jobs[job_id]["done"] = True

    threading.Thread(target=run, daemon=True).start()
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    return _jobs.get(job_id)


async def stream_job_lines(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        yield f"data: {json.dumps({'line': 'Job not found', 'done': True})}\n\n"
        return

    sent = 0
    while True:
        while sent < len(job["lines"]):
            yield f"data: {json.dumps({'line': job['lines'][sent], 'done': False})}\n\n"
            sent += 1

        if job["done"] and sent >= len(job["lines"]):
            yield f"data: {json.dumps({'line': '', 'done': True, 'exit_code': job['exit_code']})}\n\n"
            break

        await asyncio.sleep(0.1)
