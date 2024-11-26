import numpy as np
import os

class Environment():
    def __init__(self, n_states, n_rounds, state_distribution=None, utility=None):
        """
        n_states: number of states
        n_rounds: number of rounds
        state_distribution: The distribution of the states. By default, it is uniform.
        utility: The utility function the agents are using. By default, it is the equality function.
        """
        self.n_states = n_states
        self.n_rounds = n_rounds
        self.runs = []
        
        if state_distribution is None:
            self.state_distribution = np.ones(n_states) / n_states
        else:
            self.state_distribution = state_distribution

        if utility is None:
            self.utility = lambda a, b: 1 if a == b else 0
        else:
            self.utility = utility

    def run(self, sender, receiver):
        """Run the simulation for n_rounds. The simulation is as follows:
        1. Sample a state from the state distribution.
        2. The sender sends a message.
        3. The receiver plays an action.
        4. The sender receives a reward.
        5. The receiver updates its policy.
        """
        self.rewards = np.zeros(self.n_rounds)
        for t in range(self.n_rounds):
            # Sample a state
            state = np.random.choice(self.n_states, p=self.state_distribution)
            # Sender sends the message
            message = sender.send_message(state)
            # Receiver receives the message and returns the action played
            action = receiver.play_action(message)

            # Compute the reward
            reward = self.utility(state, action)

            # Update the receiver
            receiver.update(action, message, reward)

            # Update the environment
            self.rewards[t] += reward
        
        self.runs.append(np.cumsum(self.rewards))

    def dump(self, name, results_folder="results"):
        """Dump the average rewards of the simulation in a numpy file
        """
        if not self.runs:
            raise ValueError("The environment has not been run yet.")

        self.runs = np.array(self.runs)

        self.mean_reward = self.runs.mean(axis=0)
        self.std_dev = self.runs.std(axis=0)

        try:
            os.makedirs(f"{results_folder}/means")
            os.makedirs(f"{results_folder}/std_devs")
        except FileExistsError:
            pass

        np.save(f"{results_folder}/means/{name}.npy", self.rewards)
        np.save(f"{results_folder}/std_devs/{name}.npy", self.std_dev)