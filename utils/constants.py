from pathlib import Path

BASE_PATH: Path = Path(__file__).parent.parent

PID_FILE: Path = Path("/tmp/ollama_serve.pid")
OLLAMA_START: list[str] = ["ollama", "serve"]
