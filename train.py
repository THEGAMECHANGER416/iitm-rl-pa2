import numpy as np
import wandb
from env import create_standard_grid, create_four_room
from agents import TDAgent

def run_experiment():
    # Initialize a new wandb run
    wandb.init()
    cfg = wandb.config

    # 1. Initialize the Environment based on the current configuration
    if cfg.env_type == 'standard':
        env = create_standard_grid(
            start_state=np.array([cfg.start_state]),
            transition_prob=cfg.transition_prob,
            wind=cfg.wind
        )
    elif cfg.env_type == 'four_room':
        env = create_four_room(
            goal_change=cfg.goal_change,
            transition_prob=1.0
        )
    else:
        raise ValueError("Unknown environment type")

    # 2. Setup tracking for the 5 random seeds
    seeds = [42, 123, 456, 789, 1024]
    episodes = 500

    all_runs_rewards = []
    all_runs_steps = []

    # 3. Training Loop over Seeds
    for seed in seeds:
        np.random.seed(seed)

        agent = TDAgent(
            num_states=env.num_states,
            num_actions=env.num_actions,
            alpha=cfg.alpha,
            gamma=cfg.gamma,
            epsilon=cfg.epsilon if cfg.exploration == 'e-greedy' else 0.1,
            tau=cfg.tau if cfg.exploration == 'softmax' else 1.0,
            exploration=cfg.exploration
        )

        run_rewards = []
        run_steps = []

        # 4. Episode Loop
        for ep in range(episodes):
            state = env.reset()
            action = agent.choose_action(state)

            total_reward = 0
            steps = 0
            done = False

            while not done and steps < 100:
                next_state, reward = env.step(state, action)
                reward = float(reward)
                done = (next_state in env.goal_states_seq)

                if cfg.algorithm == 'sarsa':
                    next_action = agent.choose_action(next_state)
                    agent.update_sarsa(state, action, reward, next_state, next_action, done)
                    action = next_action
                elif cfg.algorithm == 'q_learning':
                    agent.update_q_learning(state, action, reward, next_state, done)
                    action = agent.choose_action(next_state)

                state = next_state
                total_reward += reward
                steps += 1

            run_rewards.append(total_reward)
            run_steps.append(steps)

        all_runs_rewards.append(run_rewards)
        all_runs_steps.append(run_steps)

    # 5. Average the metrics across all 5 seeds to get robust results
    avg_rewards = np.mean(all_runs_rewards, axis=0)
    avg_steps = np.mean(all_runs_steps, axis=0)

    # 6. Log the averaged curves to wandb
    for ep in range(episodes):
        wandb.log({
            "episode": ep,
            "avg_reward": avg_rewards[ep],
            "avg_steps": avg_steps[ep]
        })

if __name__ == "__main__":
    run_experiment()