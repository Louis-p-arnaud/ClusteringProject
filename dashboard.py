import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ANALYSIS_PATH = PROJECT_DIR / "outputs"
if not DEFAULT_ANALYSIS_PATH.exists():
    DEFAULT_ANALYSIS_PATH = PROJECT_DIR / "Algos"
ENV_ANALYSIS_PATH = os.environ.get("DASHBOARD_PATH_DATA")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dashboard with IA analysis outputs.")
    parser.add_argument(
        "-path_data",
        "--path_data",
        type=str,
        default=ENV_ANALYSIS_PATH if ENV_ANALYSIS_PATH else str(DEFAULT_ANALYSIS_PATH),
        help="Path to IA analysis outputs.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("STREAMLIT_SERVER_ADDRESS", "127.0.0.1"),
        help="Host/address for Streamlit server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("STREAMLIT_SERVER_PORT", "8501")),
        help="Port for Streamlit server.",
    )
    args = parser.parse_args()

    path_data = str(Path(args.path_data).resolve())

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_DIR / "dashboard_clustering.py"),
        f"--server.port={args.port}",
        f"--server.address={args.host}",
        "--",
        "-path_data",
        path_data,
    ]

    subprocess.run(command, cwd=str(PROJECT_DIR), check=False)


if __name__ == "__main__":
    main()
