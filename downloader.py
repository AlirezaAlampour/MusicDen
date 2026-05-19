import asyncio
import json
import os
import subprocess
import sys
import threading
import uuid
from typing import Dict, Optional

_jobs: Dict[str, dict] = {}
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _get_deemix_cmd(quality: str, output_path: str, url: str) -> list:
    """Build deemix command."""
    return [
        sys.executable, "-m", "deemix",
        "--portable",
        "-b", quality,
        "-p", output_path,
        url,
    ]


def start_download(url: str, quality: str, output_path: str, arl: str) -> str:
    """Start a deemix download subprocess. Callers must pre-validate all arguments."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"lines": [], "done": False, "exit_code": None, "process": None}

    def run():
        try:
            cmd = _get_deemix_cmd(quality, output_path, url)
        except Exception as e:
            _jobs[job_id]["lines"].append(f"[ERROR] {e}")
            _jobs[job_id]["exit_code"] = -1
            _jobs[job_id]["done"] = True
            return

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=_PROJECT_ROOT,
                bufsize=1,
            )
            _jobs[job_id]["process"] = proc

            try:
                proc.stdin.write(arl + "\n")
                proc.stdin.flush()
                proc.stdin.close()
            except Exception:
                pass

            for line in iter(proc.stdout.readline, ''):
                stripped = line.rstrip()
                if stripped:
                    _jobs[job_id]["lines"].append(stripped)
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

    yield f"data: {json.dumps({'line': '▶ deemix started...', 'done': False})}\n\n"

    sent = 0
    while True:
        while sent < len(job["lines"]):
            yield f"data: {json.dumps({'line': job['lines'][sent], 'done': False})}\n\n"
            sent += 1

        if job["done"] and sent >= len(job["lines"]):
            exit_code = job["exit_code"]
            yield f"data: {json.dumps({'line': f'Exit code: {exit_code}', 'done': True, 'exit_code': exit_code})}\n\n"
            break

        await asyncio.sleep(0.1)
