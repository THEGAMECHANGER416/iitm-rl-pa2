import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import wandb
from concurrent.futures import ProcessPoolExecutor
from env import create_standard_grid, create_four_room
from agents import TDAgent

PLOTS_DIR = "plots"
CACHE_FILE = "best_params.json"
WANDB_ENTITY = "arnavkohli321-none"
PROJECT_NAME = "RL_Assignment_2_GridWorld"
NUM_RUNS = 100
EPISODES = 500
MAX_STEPS = 100
MAX_WORKERS = 6


def fetch_best_params():
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached params from {CACHE_FILE}")
        with open(CACHE_FILE, "r") as f:
            return json.load(f)

    print("Fetching best params from wandb...")
    api = wandb.Api()
    sweeps = api.project(name=PROJECT_NAME, entity=WANDB_ENTITY).sweeps()

    best_configs = {}
    project_path = f"{WANDB_ENTITY}/{PROJECT_NAME}"
    for sweep in sweeps:
        sweep_name = sweep.config.get("name", sweep.name)
        best_run = sweep.best_run()
        if best_run is None:
            print(f"  No completed runs for sweep: {sweep_name}, skipping")
            continue

        run = api.run(f"{project_path}/{best_run.id}")
        config = dict(run.config)
        best_reward = run.summary.get("avg_reward", "N/A")
        print(f"  {sweep_name}: avg_reward = {best_reward}")
        best_configs[sweep_name] = config

    with open(CACHE_FILE, "w") as f:
        json.dump(best_configs, f, indent=2)
    print(f"Cached best params to {CACHE_FILE}")
    return best_configs


def to_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() == "true"
    return bool(val)


def create_env(cfg):
    if cfg["env_type"] == "standard":
        start = cfg["start_state"]
        if isinstance(start, str):
            start = json.loads(start)
        return create_standard_grid(
            start_state=np.array([start]),
            transition_prob=float(cfg["transition_prob"]),
            wind=to_bool(cfg["wind"])
        )
    elif cfg["env_type"] == "four_room":
        return create_four_room(
            goal_change=to_bool(cfg["goal_change"]),
            transition_prob=float(cfg.get("transition_prob", 1.0))
        )


def train_single_seed(args):
    cfg, seed = args
    np.random.seed(seed)

    env = create_env(cfg)
    exploration = cfg["exploration"]
    agent = TDAgent(
        num_states=env.num_states,
        num_actions=env.num_actions,
        alpha=float(cfg["alpha"]),
        gamma=float(cfg["gamma"]),
        epsilon=float(cfg["epsilon"]) if exploration == "e-greedy" else 0.1,
        tau=float(cfg["tau"]) if exploration == "softmax" else 1.0,
        exploration=exploration
    )

    rewards = []
    steps_list = []
    state_visits = np.zeros(env.num_states)

    for ep in range(EPISODES):
        state = env.reset()
        action = agent.choose_action(state)

        total_reward = 0
        steps = 0
        done = False

        while not done and steps < MAX_STEPS:
            state_visits[state] += 1
            next_state, reward = env.step(state, action)
            reward = float(reward)
            done = (next_state in env.goal_states_seq)

            if cfg["algorithm"] == "sarsa":
                next_action = agent.choose_action(next_state)
                agent.update_sarsa(state, action, reward, next_state, next_action, done)
                action = next_action
            elif cfg["algorithm"] == "q_learning":
                agent.update_q_learning(state, action, reward, next_state, done)
                action = agent.choose_action(next_state)

            state = next_state
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        steps_list.append(steps)

    return rewards, steps_list, state_visits, agent.Q


def plot_training_curves(all_rewards, all_steps, name, save_dir):
    avg_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    avg_steps = np.mean(all_steps, axis=0)
    std_steps = np.std(all_steps, axis=0)
    episodes = np.arange(EPISODES)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(episodes, avg_rewards, linewidth=1)
    ax1.fill_between(episodes, avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Average Reward")
    ax1.set_title(f"Reward per Episode\n{name}")
    ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, avg_steps, linewidth=1, color="tab:orange")
    ax2.fill_between(episodes, avg_steps - std_steps, avg_steps + std_steps, alpha=0.2, color="tab:orange")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Average Steps")
    ax2.set_title(f"Steps per Episode\n{name}")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}_training_curves.png"), dpi=150)
    plt.close()


def plot_state_visit_heatmap(avg_visits, env, name, save_dir):
    grid = avg_visits.reshape(env.num_rows, env.num_cols)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(grid, cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Average Visits")

    for r in range(env.num_rows):
        for c in range(env.num_cols):
            state = r * env.num_cols + c
            label = ""
            if env.obs_states is not None and any(np.all(env.obs_states == [r, c], axis=1)):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True, color="orange", alpha=0.7))
                label = "W"
            elif state in env.goal_states_seq:
                label = "G"
            elif state == env.start_state_seq:
                label = "S"
            elif env.bad_states is not None and any(np.all(env.bad_states == [r, c], axis=1)):
                label = "B"
            elif env.restart_states is not None and any(np.all(env.restart_states == [r, c], axis=1)):
                label = "R"

            if label:
                ax.text(c, r, label, ha="center", va="center", fontsize=8, fontweight="bold")

    ax.set_xticks(range(env.num_cols))
    ax.set_yticks(range(env.num_rows))
    ax.set_title(f"State Visit Heatmap\n{name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}_state_visits.png"), dpi=150)
    plt.close()


def plot_qvalue_policy(avg_q, env, name, save_dir):
    num_rows, num_cols = env.num_rows, env.num_cols
    max_q = np.max(avg_q, axis=1).reshape(num_rows, num_cols)
    best_actions = np.argmax(avg_q, axis=1).reshape(num_rows, num_cols)

    arrow_dx = [0, 0, -0.3, 0.3]
    arrow_dy = [-0.3, 0.3, 0, 0]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(max_q, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Max Q-Value")

    for r in range(num_rows):
        for c in range(num_cols):
            state = r * num_cols + c
            is_obstacle = env.obs_states is not None and any(np.all(env.obs_states == [r, c], axis=1))
            is_goal = state in env.goal_states_seq

            if is_obstacle:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True, color="orange", alpha=0.7))
            elif is_goal:
                ax.text(c, r, "G", ha="center", va="center", fontsize=8, fontweight="bold", color="white")
            else:
                a = best_actions[r, c]
                ax.arrow(c, r, arrow_dx[a], arrow_dy[a],
                         head_width=0.15, head_length=0.08, fc="black", ec="black", linewidth=0.8)

    ax.set_xticks(range(num_cols))
    ax.set_yticks(range(num_rows))
    ax.set_title(f"Q-Value Heatmap & Optimal Policy\n{name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}_qvalue_policy.png"), dpi=150)
    plt.close()


def run_config(name, cfg):
    print(f"  Running {NUM_RUNS} seeds ({MAX_WORKERS} parallel)...")

    args_list = [(cfg, seed) for seed in range(NUM_RUNS)]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(train_single_seed, args_list))

    all_rewards = np.array([r[0] for r in results])
    all_steps = np.array([r[1] for r in results])
    avg_visits = np.mean([r[2] for r in results], axis=0)
    avg_q = np.mean([r[3] for r in results], axis=0)

    # need an env instance for grid dimensions and cell annotations
    env = create_env(cfg)

    config_dir = os.path.join(PLOTS_DIR, name)
    os.makedirs(config_dir, exist_ok=True)

    plot_training_curves(all_rewards, all_steps, name, config_dir)
    plot_state_visit_heatmap(avg_visits, env, name, config_dir)
    plot_qvalue_policy(avg_q, env, name, config_dir)
    print(f"  Plots saved to {config_dir}/")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    best_configs = fetch_best_params()

    for name, cfg in best_configs.items():
        print(f"\n{'='*50}")
        print(f"Config: {name}")
        print(f"{'='*50}")
        run_config(name, cfg)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
