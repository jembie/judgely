import atexit
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from utils.config import OLLAMA_START, PID_FILE


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

    process = subprocess.Popen(
        OLLAMA_START,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

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


def _wait_for_port(
    host: str = "localhost",
    port: int = 11434,
    timeout: float = 10.0,
    interval: float = 0.1,
):
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection(address=(host, port), timeout=1):
                return
        except OSError:
            pass

        if time.monotonic() > deadline:
            raise TimeoutError(f"Port {port} not open after {timeout}s")

        time.sleep(interval)


def launch():
    ollama_process = start_ollama()

    if ollama_process:
        # Currently we are too fast until the Connection is fully established, hence we sleep to assure that it is connected
        _wait_for_port()
        atexit.register(lambda: kill_ollama(ollama_process))
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, _singal_handler)
