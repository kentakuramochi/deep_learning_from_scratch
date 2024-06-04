import matplotlib.pyplot as plt
import numpy as np

from bandit import Agent


class NonStatBandit:
    """Multi-armed bandit on non-stationary problem.

    Attributes:
        arms (int): Number of the bandits.
        rates (numpy.ndarray[float]): Winning rates of each one-armed bandits (slot machines).
    """

    def __init__(self, arms=10):
        """Initialize one-armed bandits with random winning rates.

        Args:
            arms (int): Number of the bandits.
        """
        self.arms = arms
        self.rates = np.random.rand(arms)

    def play(self, arm):
        """Play a one-armed bandit.

        Args:
            arm (int): Index of the bandit.

        Returns:
            (int): 1 if win, 0 otherwise.
        """
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.randn(self.arms)  # Add noise
        if rate > np.random.rand():
            return 1
        else:
            return 0


class AlphaAgent:
    """Agent by epsilon-greedy method for the non-stationary multi-armed bandit problem.

    Attributes:
        epsilon (float): Probability of the exploration.
        Qs (numpy.ndarray[float]): Estimated action values for each action.
        alpha (float): Coefficient of the exponential moving average for action values.
    """

    def __init__(self, epsilon, alpha, action_size=10):
        """Initialize an agent.

        Args:
            epsilon (float): Probability of the exploration.
            alpha (float): Coefficient of the exponential moving average for action values.
            action_size (int): Number of actions.
        """
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.alpha = alpha

    def update(self, action, reward):
        """Update the estimated action values.

        Args:
            action (int): Action of the agent.
            reward (float): Current reward.
        """
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self):
        """Get an action by action values.

        Returns:
            (int): Action of the agent.
        """
        if np.random.rand() < self.epsilon:  # Action randomly by the probability
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)


runs = 200  # Number of experiments
steps = 1000
epsilon = 0.1
alpha = 0.8
all_rates = np.zeros((2, runs, steps))

for i, agent in enumerate((Agent(epsilon), AlphaAgent(epsilon, alpha))):
    for run in range(runs):
        bandit = NonStatBandit()
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))

        all_rates[i, run] = rates

    # Average winning rates for each step
    avg_rates = np.average(all_rates, axis=1)

# Plot the winning rates
plt.ylabel("Rates")
plt.xlabel("Steps")
plt.plot(avg_rates[0], label="sample average")
plt.plot(avg_rates[1], label="constant alpha")
plt.legend()
plt.savefig("non_stationary_bandit_200_avg_rates.png")
