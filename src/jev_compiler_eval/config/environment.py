"""Environment loading used only by command-line entry points."""

from pathlib import Path

from dotenv import load_dotenv


def load_cli_environment(path: Path) -> None:
    """Load a CLI environment file without overriding exported values."""

    load_dotenv(path, override=False)
