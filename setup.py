"""
Quick Setup Script — Get the recommendation service running in one command.

Usage:
    python setup.py

What it does:
    1. Install Python dependencies
    2. Verify MovieLens-100k data exists (download if missing)
    3. Seed database (SQLite)
    4. Train SVD model
    5. Print instructions to start the API server
"""
import os
import sys
import subprocess
import shutil
import zipfile
import urllib.request


MOVIE_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MOVIE_AGENT_DIR, "data", "ml-100k")
DB_PATH = os.path.join(MOVIE_AGENT_DIR, "data", "movies.db")
MODEL_PATH = os.path.join(MOVIE_AGENT_DIR, "models", "svd_model.pkl")
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def run_cmd(cmd: list, cwd: str = MOVIE_AGENT_DIR):
    """Run a command and print output."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        print(f"  ❌ Command failed with exit code {result.returncode}")
        sys.exit(1)


def step_install_deps():
    """Step 1: Install Python dependencies."""
    print("\n" + "=" * 60)
    print("Step 1/4: Installing dependencies...")
    print("=" * 60)

    req_file = os.path.join(MOVIE_AGENT_DIR, "requirements.txt")
    run_cmd([sys.executable, "-m", "pip", "install", "-r", req_file, "-q"])
    print("  Dependencies installed")


def step_check_data():
    """Step 2: Verify MovieLens-100k data exists."""
    print("\n" + "=" * 60)
    print("Step 2/4: Checking MovieLens-100k data...")
    print("=" * 60)

    required_files = ["u.data", "u.item", "u.user"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(DATA_DIR, f))]

    if not missing:
        print("  MovieLens-100k data found")
        return

    print(f"  Missing files: {missing}")
    print(f"  Downloading MovieLens-100k from {MOVIELENS_URL}...")

    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(MOVIE_AGENT_DIR, "data", "ml-100k.zip")

    try:
        urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
        print("  Extracting...")

        with zipfile.ZipFile(zip_path, "r") as z:
            # Extract to temp then move
            temp_dir = os.path.join(MOVIE_AGENT_DIR, "data", "_temp_ml")
            z.extractall(temp_dir)

            # Move files from extracted ml-100k folder
            extracted_dir = os.path.join(temp_dir, "ml-100k")
            if os.path.exists(extracted_dir):
                for f in os.listdir(extracted_dir):
                    src = os.path.join(extracted_dir, f)
                    dst = os.path.join(DATA_DIR, f)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)

            shutil.rmtree(temp_dir, ignore_errors=True)

        os.remove(zip_path)
        print("  MovieLens-100k downloaded and extracted")

    except Exception as e:
        print(f"  Download failed: {e}")
        print(f"  Manual download: {MOVIELENS_URL}")
        print(f"     Extract to: {DATA_DIR}")
        sys.exit(1)


def step_seed_db():
    """Step 3: Seed SQLite database."""
    print("\n" + "=" * 60)
    print("Step 3/4: Seeding database...")
    print("=" * 60)

    run_cmd([
        sys.executable, "-m", "pipeline.seed_database",
        "--target", "sqlite",
        "--db-path", DB_PATH,
    ])


def step_train():
    """Step 4: Train SVD model."""
    print("\n" + "=" * 60)
    print("Step 4/4: Training SVD model...")
    print("=" * 60)

    run_cmd([
        sys.executable, "-m", "pipeline.run_pipeline",
        "--data-source", "database",
        "--db-path", DB_PATH,
        "--skip-benchmark",
        "--skip-gridsearch",
        "--no-redis",
    ])


def main():
    print("=" * 60)
    print("Movie Recommendation Service — Quick Setup")
    print("=" * 60)
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  Dir:     {MOVIE_AGENT_DIR}")

    step_install_deps()
    step_check_data()
    step_seed_db()
    step_train()

    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("  Start the API server:")
    print("    python -m app.main")
    print()
    print("  Then open:")
    print("    http://localhost:8000/docs          — Swagger UI")
    print("    http://localhost:8000/health         — Health check")
    print("    http://localhost:8000/api/v1/recommendations/1  — Test")
    print()


if __name__ == "__main__":
    main()
