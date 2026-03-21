import numpy as np

class TDAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1, tau=1.0, exploration='e-greedy'):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau
        self.exploration = exploration
        
        # Initialize Q-table with zeros
        # The environment uses 1D state representations (0 to num_states-1) [cite: 139, 152]
        self.Q = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        """Selects an action based on the current exploration strategy."""
        if self.exploration == 'e-greedy':
            if np.random.rand() < self.epsilon:
                # Explore: random action
                return np.random.choice(self.num_actions)
            else:
                # Exploit: choose best action, break ties randomly to prevent getting stuck
                q_values = self.Q[state]
                max_q = np.max(q_values)
                return np.random.choice(np.where(q_values == max_q)[0])
                
        elif self.exploration == 'softmax':
            # Softmax exploration with temperature tau
            q_values = self.Q[state]
            # Subtract max for numerical stability before applying exp()
            preferences = q_values / self.tau
            max_pref = np.max(preferences)
            exp_preferences = np.exp(preferences - max_pref)
            action_probs = exp_preferences / np.sum(exp_preferences)
            
            return np.random.choice(self.num_actions, p=action_probs)
        else:
            raise ValueError("Unknown exploration strategy. Use 'e-greedy' or 'softmax'.")

    def update_q_learning(self, state, action, reward, next_state, done):
        """Off-policy update: Uses the maximum Q-value of the next state."""
        # If the episode is done, there is no future reward
        best_next_q = np.max(self.Q[next_state]) if not done else 0.0
        
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def update_sarsa(self, state, action, reward, next_state, next_action, done):
        """On-policy update: Uses the Q-value of the next state-action pair."""
        # If the episode is done, there is no future reward
        next_q = self.Q[next_state, next_action] if not done else 0.0
        
        td_target = reward + self.gamma * next_q
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        
    def reset_q_table(self):
        """Clears the Q-table for a fresh training run."""
        self.Q = np.zeros((self.num_states, self.num_actions))