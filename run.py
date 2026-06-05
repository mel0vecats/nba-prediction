import subprocess
import sys

def run_script(script_path):
    print(f"\n=== Running {script_path} ===\n")
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    if result.returncode != 0:
        print(f"Error running {script_path}")
        sys.exit(1)

if __name__ == "__main__":
    scripts = [
        "src/get_teams.py",
        "src/fetch_historical.py",
        "src/fetch_upcoming.py",
        "src/feature_engineering.py",
        "src/train.py",
        "src/predict.py",
    ]

    for script in scripts:
        run_script(script)

    print("\n✅ All scripts executed successfully!")