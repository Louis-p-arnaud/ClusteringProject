import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def run_pipeline(script_path: Path, data_path: Path) -> None:
    env = os.environ.copy()
    env["DATA_PATH"] = str(data_path)

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_DIR),
        env=env,
        check=True,
    )


def copy_outputs(source_output: Path, algo_key: str, output_root: Path) -> None:
    if not source_output.exists():
        return

    output_root.mkdir(parents=True, exist_ok=True)

    for xlsx_file in source_output.glob("*.xlsx"):
        # Copy into root output folder with collision-safe naming
        if xlsx_file.name == "save_metric.xlsx":
            root_name = f"save_metric_{algo_key}.xlsx"
        else:
            root_name = xlsx_file.name

        root_file = output_root / root_name
        if xlsx_file.resolve() != root_file.resolve():
            shutil.copy2(xlsx_file, root_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all IA clustering pipelines and consolidate outputs.")
    parser.add_argument(
        "-path_data",
        "--path_data",
        type=str,
        default=str(PROJECT_DIR / "data" / "test"),
        help="Path to dataset images.",
    )
    parser.add_argument(
        "-path_output",
        "--path_output",
        type=str,
        default=str(PROJECT_DIR / "outputs"),
        help="Path to output folder where analyses are consolidated.",
    )
    args = parser.parse_args()

    data_path = Path(args.path_data).resolve()
    output_root = Path(args.path_output).resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    pipelines = [
        {
            "algo_key": "kmeans",
            "algo_folder": "kmeans_algo",
            "script": PROJECT_DIR / "Algos" / "kmeans_algo" / "pipeline_kmeans.py",
            "source_output": PROJECT_DIR / "Algos" / "kmeans_algo" / "output",
        },
        {
            "algo_key": "dbscan",
            "algo_folder": "dbscan_algo",
            "script": PROJECT_DIR / "Algos" / "dbscan_algo" / "pipeline_dbscan.py",
            "source_output": PROJECT_DIR / "Algos" / "dbscan_algo" / "output",
        },
        {
            "algo_key": "spectral",
            "algo_folder": "spectral_clustering_algo",
            "script": PROJECT_DIR / "Algos" / "spectral_clustering_algo" / "pipeline_spectral.py",
            "source_output": PROJECT_DIR / "Algos" / "spectral_clustering_algo" / "output",
        },
        {
            "algo_key": "gmm",
            "algo_folder": "GMM_algo",
            "script": PROJECT_DIR / "Algos" / "GMM_algo" / "pipeline_GMM.py",
            "source_output": PROJECT_DIR / "Algos" / "GMM_algo" / "output",
        },
    ]

    print("\n##### Run Pipeline IA #####")
    print(f"- path_data   : {data_path}")
    print(f"- path_output : {output_root}")

    # Cleanup legacy folder-based copies from previous versions.
    for legacy_folder in ["kmeans_algo", "dbscan_algo", "spectral_clustering_algo", "GMM_algo"]:
        legacy_path = output_root / legacy_folder
        if legacy_path.exists() and legacy_path.is_dir():
            shutil.rmtree(legacy_path)

    for cfg in pipelines:
        print(f"\n--- Running {cfg['algo_key'].upper()} pipeline ---")
        run_pipeline(cfg["script"], data_path)
        copy_outputs(cfg["source_output"], cfg["algo_key"], output_root)

    print("\nDone. Consolidated outputs are available in:")
    print(output_root)


if __name__ == "__main__":
    main()
