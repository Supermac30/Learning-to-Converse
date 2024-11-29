import hydra

from LearningToConverse.utils import build_prior, build_utility, build_sender, build_receiver
from LearningToConverse.environment import Environment
import numpy as np

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    np.random.seed(cfg.seed)

    # Initialize the sender & receiver
    sender = build_sender(cfg)
    receiver = build_receiver(cfg)
    # Initialize the environment
    prior = build_prior(cfg)
    utility = build_utility(cfg)
    env = Environment(cfg.n_states, cfg.n_rounds, prior, utility)

    # Run the simulation
    for _ in range(cfg.n_simulations):
        env.run(sender, receiver)
    # Dump the results
    env.dump(f"{sender.name}_{receiver.name}")


if __name__ == "__main__":
    main()