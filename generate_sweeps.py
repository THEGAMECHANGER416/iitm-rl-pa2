import os

SWEEP_DIR = "sweep_configs"

def create_yaml_string(run_name, env_type, algo, trans_prob, start_state, wind, goal_change, exploration):
    yaml = f"""program: train.py
method: grid
name: {run_name}
metric:
  name: avg_reward
  goal: maximize
parameters:
  env_type:
    value: "{env_type}"
  algorithm:
    value: "{algo}"
  transition_prob:
    value: {trans_prob}
  start_state:
    value: {start_state}
  wind:
    value: {wind}
  goal_change:
    value: {goal_change}
  exploration:
    value: "{exploration}"
  alpha:
    values: [0.001, 0.01, 0.1, 1.0]
  gamma:
    values: [0.7, 0.8, 0.9, 1.0]
"""
    if exploration == 'e-greedy':
        yaml += """  epsilon:
    values: [0.001, 0.01, 0.05, 0.1]
  tau:
    value: 1.0
"""
    elif exploration == 'softmax':
        yaml += """  epsilon:
    value: 0.1
  tau:
    values: [0.01, 0.1, 1.0, 2.0]
"""
    return yaml

def main():
    os.makedirs(SWEEP_DIR, exist_ok=True)
    config_count = 1
    
    # 1. Standard Grid - Q-Learning (8 Configs)
    for trans in [0.7, 1.0]:
        for start in ["[0, 4]", "[3, 6]"]:
            for exp in ["e-greedy", "softmax"]:
                name = f"C{config_count}_Standard_QL_trans{trans}_start{start.replace(' ', '').replace('[', '').replace(']', '')}_{exp}"
                yaml_content = create_yaml_string(name, "standard", "q_learning", trans, start, "False", "False", exp)
                with open(os.path.join(SWEEP_DIR, f"sweep_config_{config_count}.yaml"), "w") as f:
                    f.write(yaml_content)
                config_count += 1

    # 2. Standard Grid - SARSA (8 Configs)
    for wind in ["True", "False"]:
        for start in ["[0, 4]", "[3, 6]"]:
            for exp in ["e-greedy", "softmax"]:
                name = f"C{config_count}_Standard_SARSA_wind{wind}_start{start.replace(' ', '').replace('[', '').replace(']', '')}_{exp}"
                yaml_content = create_yaml_string(name, "standard", "sarsa", 1.0, start, wind, "False", exp)
                with open(os.path.join(SWEEP_DIR, f"sweep_config_{config_count}.yaml"), "w") as f:
                    f.write(yaml_content)
                config_count += 1

    # 3. Four Room Grid - Q-Learning (2 Configs)
    for goal_change in ["True", "False"]:
        name = f"C{config_count}_FourRoom_QL_goalchange{goal_change}"
        yaml_content = create_yaml_string(name, "four_room", "q_learning", 1.0, "[8, 0]", "False", goal_change, "e-greedy")
        with open(os.path.join(SWEEP_DIR, f"sweep_config_{config_count}.yaml"), "w") as f:
            f.write(yaml_content)
        config_count += 1

    # 4. Four Room Grid - SARSA (2 Configs)
    for goal_change in ["True", "False"]:
        name = f"C{config_count}_FourRoom_SARSA_goalchange{goal_change}"
        yaml_content = create_yaml_string(name, "four_room", "sarsa", 1.0, "[8, 0]", "False", goal_change, "e-greedy")
        with open(os.path.join(SWEEP_DIR, f"sweep_config_{config_count}.yaml"), "w") as f:
            f.write(yaml_content)
        config_count += 1

    print(f"Generated {config_count - 1} sweep configs in {SWEEP_DIR}/")

if __name__ == "__main__":
    main()