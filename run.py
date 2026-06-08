import subprocess
import sys

scripts = [
    "src/get_teams.py",
    "src/fetch_historical.py",
    "src/fetch_upcoming.py",
    "src/feature_engineering.py",
    "src/train.py",
    "src/predict.py"
]

def run_script(script):
    print(f"\nRunning {script}\n", flush=True)

    process = subprocess.Popen(
        [sys.executable, "-u", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # stream logs live
    for line in process.stdout:
        print(line, end="", flush=True)

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"{script} failed with exit code {process.returncode}")

for script in scripts:
    run_script(script)

print("\nAll scripts completed successfully!", flush=True)