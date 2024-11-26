import numpy as np

class Receiver():
    """The agent that receives the message from the sender and must play an action."""
    def __init__(self, n_states, n_messages):
        self.n_states = n_states
        self.n_messages = n_messages
        self.name = "Unspecified Receiver"

    def play_action(self, message):
        raise NotImplementedError
    
    def update(self, action, message, reward):
        raise NotImplementedError


class Exp3(Receiver):
    """The receiver that uses multiplicative weights to play the contextual bandit that the sender is inducing.
    """
    def __init__(self, n_states, n_messages, n_rounds, eta):
        super().__init__(n_states, n_messages)
        self.eta = eta * np.sqrt(np.log(n_states) / (n_states * n_rounds))

        self.total_reward = np.zeros((n_states, n_messages))
        self.times_seen = np.zeros((n_states, n_messages))

        self.name = f"Exp3"

    def play_action(self, message):
        """Play the action that maximizes the expected reward.
        """
        total_reward = self.total_reward[:, message]
        weights = np.exp(self.eta * total_reward - max(self.eta * total_reward))
        normalized_weights = weights / weights.sum()
        return np.random.choice(self.n_states, p=normalized_weights)
    
    def update(self, action, message, reward):
        """Update the weights of the receiver.
        """
        self.total_reward[action, message] += reward


class UCB(Receiver):
    """The receiver that uses the Upper Confidence Bound algorithm to play the contextual bandit that the sender is inducing.
    """
    def __init__(self, n_states, n_messages, n_rounds, delta_exponent):
        super().__init__(n_states, n_messages)
        self.delta_exponent = delta_exponent
        self.n = n_rounds

        self.times_seen = np.zeros((n_states, n_messages))
        self.mean_rewards = np.zeros((n_states, n_messages))

        self.name = f"UCB"

    def play_action(self, message):
        """Play the action that maximizes the expected reward.
        """
        times_seen = self.times_seen[:, message]
        mean_rewards = self.mean_rewards[:, message]
        with np.errstate(divide='ignore', invalid='ignore'):
            ucb = mean_rewards + np.sqrt((self.delta_exponent * np.log(self.n)) / times_seen)
            ucb[times_seen == 0] = np.inf

        return np.argmax(ucb)
    
    def update(self, action, message, reward):
        """Update the mean rewards and the times seen of the receiver.
        """
        self.times_seen[action, message] += 1
        self.mean_rewards[action, message] += (reward - self.mean_rewards[action, message]) / self.times_seen[action, message]