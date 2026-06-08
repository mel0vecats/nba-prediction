import subprocess
import sys

scripts = [
    "src/get_teams.py",
    "src/fetch_historical.py",
    "src/fetch_upcoming.py",
    "src/feature_engineering.py",
    "src/train.py",
    "src/predict.py",
]

for script in scripts:
    print(f"\nRunning {script}...")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(f"Errors in {script}:\n{result.stderr}")
    
    if result.returncode != 0:
        print(f"{script} failed with exit code {result.returncode}. Stopping.")
        sys.exit(result.returncode)

print("\nAll scripts completed successfully.")