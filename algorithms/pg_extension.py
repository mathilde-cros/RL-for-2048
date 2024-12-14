import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

# We created this alternative module to integrate pre-trained agent's weights into the policy gradient algorithm


class PolicyNetworkCNN(nn.Module):
    def __init__(self):
        super(PolicyNetworkCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def load_pretrained_weights(self, weights):
        self.load_state_dict(weights)


class PolicyGradientStrategy:
    def __init__(self, hidden_sizes=[128], learning_rate=1e-3, activation_fn=nn.ReLU, optimizer_cls=optim.Adam, gamma=0.99, entropy_coef=0.0):
        self.actions = ['up', 'down', 'left', 'right']
        self.policy_net = PolicyNetworkCNN()
        self.optimizer = optimizer_cls(
            self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.saved_log_probs = []
        self.rewards = []

    def preprocess_state(self, state):
        state = np.array(state, dtype=np.float32).reshape(1, 1, 4, 4)
        return torch.tensor(state, dtype=torch.float32)

    def select_action(self, state):
        state_tensor = self.preprocess_state(state)
        action_logits = self.policy_net(state_tensor)
        action_probs = torch.softmax(action_logits, dim=-1)
        m = torch.distributions.Categorical(action_probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return self.actions[action.item()]

    def train(self):
        R = 0
        policy_loss = []
        returns = []

        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        baseline = returns.mean()
        for log_prob, R in zip(self.saved_log_probs, returns):
            advantage = R - baseline
            policy_loss.append(-log_prob * advantage)

        entropy_loss = -self.entropy_coef * \
            torch.stack(self.saved_log_probs).mean()
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + entropy_loss
        loss.backward()
        self.optimizer.step()

        self.saved_log_probs = []
        self.rewards = []

    def load_pretrained_policy(self, weights):
        self.policy_net.load_pretrained_weights(weights)

    def __call__(self, state):
        return self.select_action(state)

# In order to run a set of experiments with different configurations of the actor/critic network, we again defined a separate class to more efficiently automate the experiment running/testing


class ExperimentPolicyGradient:
    def __init__(self, input_size=16, output_size=4, learning_rate=1e-3, gamma=0.99, entropy_coef=0.0):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef

    def run_experiment(self, hidden_layers_configs, num_trials):
        results = {}
        for config in hidden_layers_configs:
            scores = []
            for _ in range(num_trials):
                strategy = PolicyGradientStrategy(hidden_sizes=config, learning_rate=self.learning_rate,
                                                  gamma=self.gamma, entropy_coef=self.entropy_coef)

                total_score = 0
                for _ in range(100):
                    # Simulate a random state and reward environment
                    state = np.random.rand(4, 4)
                    action = strategy(state)
                    reward = np.random.uniform(-1, 1)  # Simulated reward
                    strategy.rewards.append(reward)
                    total_score += reward
                strategy.train()
                scores.append(total_score)
            avg_score = sum(scores) / len(scores)
            results[tuple(config)] = avg_score

        results_df = pd.DataFrame({
            "Configuration": [str(config) for config in results.keys()],
            "Average Score": list(results.values())
        })
        results_df.to_csv(
            "./results/policy_gradient_experiment_results.csv", index=False)
        print(
            "Results saved to policy_gradient_experiment_results.csv in the results folder")

        # Visualize results
        configs = [str(config) for config in results.keys()]
        scores = list(results.values())

        plt.figure(figsize=(10, 6))
        plt.bar(configs, scores)
        plt.xlabel("Network Configuration")
        plt.ylabel("Average Score")
        plt.title("Performance vs Network Configuration")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("./results/policy_gradient_experiment_results.png")
        plt.show()
        print("Results visualized and saved as policy_gradient_experiment_results.png in the results folder")

        return results


if __name__ == "__main__":
    hidden_layers_configs = [[64], [128], [256], [128, 128], [256, 128], [256, 256], [512, 256, 128], [512, 512, 256], [512, 512, 512], [1024, 512, 256], [1024, 1024, 512], [1024, 1024, 1024], [2048, 1024, 512], [2048, 2048, 1024], [2048, 2048, 2048], [4096, 2048, 1024], [4096, 4096, 2048], [4096, 4096, 4096], [8192, 4096, 2048], [8192, 8192, 4096], [8192, 8192, 8192], [16384, 8192, 4096], [16384, 16384, 8192], [16384, 16384, 16384], [
        32768, 16384, 8192], [32768, 32768, 16384], [32768, 32768, 32768], [65536, 32768, 16384], [65536, 65536, 32768], [65536, 65536, 65536], [131072, 65536, 32768], [131072, 131072, 65536], [131072, 131072, 131072], [262144, 131072, 65536], [262144, 262144, 131072], [262144, 262144, 262144], [524288, 262144, 131072], [524288, 524288, 262144], [524288, 524288, 524288], [1048576, 524288, 262144], [1048576, 1048576, 524288], [1048576, 1048576, 1048576]]
    num_trials = 10

    pretrained_weights_path = "./data/policy_network_best.pth"
    pretrained_model = PolicyNetworkCNN()
    pretrained_model.load_state_dict(torch.load(pretrained_weights_path))

    strategy = PolicyGradientStrategy()
    strategy.load_pretrained_policy(pretrained_model.state_dict())

    experiment = ExperimentPolicyGradient()
    results = experiment.run_experiment(hidden_layers_configs, num_trials)
    print("Experimental Results:", results)
