import numpy as np

class Sender():
    """The agent that has all the information and sends messages to the receiver."""
    def __init__(self, n_states, n_messages):
        self.n_states = n_states
        self.n_messages = n_messages

        self.name = "Unspecified Sender"

    def send_message(self, state):
        raise NotImplementedError
    

class ThresholdSender(Sender):
    """The sender that only swaps to different messaging schemes
    when the regret from not switching is above a certain threshold.
    """
    def __init__(self, n_states, n_messages, threshold):
        super().__init__(n_states, n_messages)
        self.threshold = threshold

        # The ordering of states from most to least common, initilized randomly
        self.ordering = np.random.permutation(n_states)
        # A pointer for quick indexing
        self.pointer = np.argsort(self.ordering)

        # The frequency of each state
        self.times_seen = np.zeros(n_states)
        self.total_time = 0

        self.name = f"Threshold={threshold}"

    def send_message(self, state):
        self.total_time += 1
        self.times_seen[state] += 1

        index = self.pointer[state]
        while index > 0 and self.times_seen[state] > self.times_seen[self.ordering[index - 1]] + self.threshold * self.total_time:
            # Swap the ordering to have the more common state first
            self.ordering[index], self.ordering[index - 1] = self.ordering[index - 1], self.ordering[index]
            self.pointer[self.ordering[index]] = index
            self.pointer[self.ordering[index - 1]] = index - 1
            index -= 1

        if index < self.n_messages:
            return index
        else:
            return self.n_messages - 1
