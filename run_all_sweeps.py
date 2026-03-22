import yaml
import wandb
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from train import run_experiment

SWEEP_DIR = "sweep_configs"
PROJECT_NAME = "RL_Assignment_2_GridWorld"
MAX_WORKERS = 4

def run_sweep(i):
    yaml_file = os.path.join(SWEEP_DIR, f"sweep_config_{i}.yaml")

    if not os.path.exists(yaml_file):
        print(f"Config not found: {yaml_file}, skipping...")
        return i, False

    print(f"[Sweep {i}/20] Starting: {yaml_file}")

    with open(yaml_file, 'r') as file:
        sweep_config = yaml.safe_load(file)

    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
    wandb.agent(sweep_id, function=run_experiment, project=PROJECT_NAME)

    print(f"[Sweep {i}/20] Done.")
    return i, True

def main():
    wandb.login()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_sweep, i): i for i in range(1, 21)}

        for future in as_completed(futures):
            i, success = future.result()
            if not success:
                print(f"Sweep {i} skipped.")

    print("\nAll sweeps completed.")

if __name__ == "__main__":
    main()