import numpy as np
from LearningToConverse.sender import ThresholdSender
from LearningToConverse.receiver import Exp3, UCB

def build_prior(cfg):
    if cfg.prior_name == "uniform":
        prior = np.ones(cfg.n_states) / cfg.n_states
    elif cfg.prior_name == "skewed":
        # Return the prior with a random element set to probability 0.75 and the rest normalized
        skewed_element = np.random.randint(cfg.n_states)
        prior = np.ones(cfg.n_states) * (0.25 / (cfg.n_states - 1))
        prior[skewed_element] = 0.75
    elif cfg.prior_name == "gaussian":
        mean = np.random.uniform(0, 1)
        std_dev = np.random.uniform(0.1, 0.5)
        x = np.linspace(0, 1, cfg.n_states)
        prior = np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
        prior /= prior.sum()  # Normalize to make it a valid probability distribution
    elif cfg.prior_name == "random":
        prior = np.random.random(cfg.n_states)
        prior /= prior.sum()
    else:
        raise ValueError("Invalid prior name.")
    
    return prior


def build_utility(cfg):
    if cfg.utility_name == "equality":
        utility = lambda a, b: 1 if a == b else 0
    elif cfg.utility_name == "inequality":
        utility = lambda a, b: 1 if a != b else 0
    else:
        raise ValueError("Invalid utility name.")
    
    return utility


def build_sender(cfg):
    if cfg.sender_name == "threshold":
        sender = ThresholdSender(cfg.n_states, cfg.n_messages, cfg.threshold)
    else:
        raise ValueError("Invalid sender name.")
    
    return sender


def build_receiver(cfg):
    if cfg.receiver_name == "exp3":
        receiver = Exp3(cfg.n_states, cfg.n_messages, cfg.n_states, cfg.eta)
    elif cfg.receiver_name == "ucb":
        receiver = UCB(cfg.n_states, cfg.n_messages, cfg.n_states, cfg.delta_exponent)
    else:
        raise ValueError("Invalid receiver name.")
    
    return receiver