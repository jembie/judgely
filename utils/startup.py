import subprocess
import os
import signal
import atexit
import sys
from pathlib import Path
from utils.config import PID_FILE, OLLAMA_START
import time


def is_processing_running(pid: int) -> bool:
    """
    Args:
        pid (int): Process Identifier to be inspected

    Returns:
        bool: Returns True if there's a process with the input `PID`
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False

    return True


def start_ollama() -> subprocess.Popen | None:
    global PID_FILE

    PID_FILE = Path(PID_FILE)
    if PID_FILE.exists():

        try:
            current_pid = int(PID_FILE.read_text())
            if is_processing_running(current_pid):
                print(f"[ollama] is already running (PID: {current_pid}); skipping initialisation")
                return None
        except Exception:
            pass

    process = subprocess.Popen(OLLAMA_START, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)

    PID_FILE.write_text(str(process.pid))
    return process


def kill_ollama(process: subprocess.Popen | None):
    if not process:
        return

    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGTERM)
        print(f"[ollama] sent SIGTERM to PGID {pgid}")
    except Exception as e:
        print(f"[ollama] error terminating: {e!r}")

    finally:
        try:
            PID_FILE.unlink()
        except FileNotFoundError:
            pass


def _singal_handler(signal_number, frame):
    sys.exit(0)


def launch():
    ollama_process = start_ollama()

    if ollama_process:
        # Currently we are too fast until the Connection is fully established, hence we sleep to assure that it is connected
        time.sleep(3)
        atexit.register(lambda: kill_ollama(ollama_process))
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, _singal_handler)
