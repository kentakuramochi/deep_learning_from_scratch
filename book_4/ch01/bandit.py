import matplotlib.pyplot as plt
import numpy as np


class Bandit:
    """Multi-armed bandit.

    Attributes:
        rates (ndarray[float]): Winning rates of each one-armed bandits (slot machines).
    """

    def __init__(self, arms=10):
        """Initialize one-armed bandits with random winning rates.

        Args:
            args (int): Number of the bandits.
        """
        self.rates = np.random.rand(arms)

    def play(self, arm):
        """Play a one-armed bandit.

        Args:
            arm (int): Index of the bandit.

        Returns:
            (int): 1 if win, 0 otherwise.
        """
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0


class Agent:
    """Agent by epsilon-greedy method for the multi-armed bandit problem.

    Attributes:
        epsilon (float): Probability of the exploration.
        Qs (ndarray[float]): Estimated action values for each action.
        ns (ndarray[int]): Number of each action.
    """

    def __init__(self, epsilon, action_size=10):
        """Initialize an agent.

        Args:
            epsilon (float): Probability of the exploration.
            action_size (int): Number of actions.
        """
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        """Update the estimated action values.

        Args:
            action (int): Action of the agent.
            reward (float): Current reward.
        """
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        """Get an action by action values.

        Returns:
            (int): Action of the agent.
        """
        if np.random.rand() < self.epsilon:  # Action randomly by the probability
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)


def main():
    # bandit = Bandit()
    #
    # Qs = np.zeros(10)
    # ns = np.zeros(10)
    #
    # for n in range(10):
    #     action = np.random.randint(0, 10)  # Random action
    #     reward = bandit.play(action)
    #
    #     ns[action] += 1
    #     Qs[action] += (reward - Qs[action]) / ns[action]
    #     print(Qs)

    # steps = 1000
    # epsilon = 0.1
    #
    # bandit = Bandit()
    # agent = Agent(epsilon)
    #
    # total_reward = 0
    # total_rewards = []
    # rates = []
    #
    # for step in range(steps):
    #     action = agent.get_action()  # Get an action
    #     reward = bandit.play(action)  # Play a bandit and get the reward
    #     agent.update(action, reward)  # Update
    #     total_reward += reward
    #
    #     total_rewards.append(total_reward)
    #     rates.append(total_reward / (step + 1))
    #
    # print(total_reward)
    #
    # # Plot total rewards
    # plt.ylabel("Total reward")
    # plt.xlabel("Steps")
    # plt.plot(total_rewards)
    # plt.savefig("bandit_total_rewards.png")
    #
    # plt.cla()
    #
    # # Plot winning rates
    # plt.ylabel("Rates")
    # plt.xlabel("Steps")
    # plt.plot(rates)
    # plt.savefig("bandit_rates.png")

    # runs = 200  # Number of experiments
    # steps = 1000
    # epsilon = 0.1
    # all_rates = np.zeros((runs, steps))
    #
    # for run in range(runs):
    #     bandit = Bandit()
    #     agent = Agent(epsilon)
    #     total_reward = 0
    #     rates = []
    #
    #     for step in range(steps):
    #         action = agent.get_action()
    #         reward = bandit.play(action)
    #         agent.update(action, reward)
    #         total_reward += reward
    #         rates.append(total_reward / (step + 1))
    #
    #     all_rates[run] = rates
    #
    # # Average winning rates for each step
    # avg_rates = np.average(all_rates, axis=0)
    #
    # # Plot the winning rates
    # plt.ylabel("Rates")
    # plt.xlabel("Steps")
    # plt.plot(avg_rates)
    # plt.savefig("bandit_200_avg_rates.png")

    runs = 200
    steps = 1000
    epsilons = [0.1, 0.3, 0.01]
    all_rates = np.zeros((len(epsilons), runs, steps))

    for i, epsilon in enumerate(epsilons):  # Experiment by epsilons
        for run in range(runs):
            bandit = Bandit()
            agent = Agent(epsilon)
            total_reward = 0
            rates = []

            for step in range(steps):
                action = agent.get_action()
                reward = bandit.play(action)
                agent.update(action, reward)
                total_reward += reward
                rates.append(total_reward / (step + 1))

            all_rates[i, run] = rates

        avg_rates = np.average(all_rates, axis=1)

    # Plot the winning rates
    plt.ylabel("Rates")
    plt.xlabel("Steps")
    for i in range(len(epsilons)):
        plt.plot(avg_rates[i], label=f"{epsilons[i]:.2f}")
    plt.legend()
    plt.savefig("bandit_by_epsilons.png")


if __name__ == "__main__":
    main()
