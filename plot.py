import matplotlib.pyplot as plt
import numpy as np
import os

import hydra


def line_break(string, max_length):
    """Insert a line break in a string every max_length characters.
    """
    words = string.split()
    new_string = ""
    line_length = 0
    for word in words:
        if line_length + len(word) > max_length:
            new_string += "\n"
            line_length = 0
        new_string += word + " "
        line_length += len(word) + 1
    return new_string


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    # Plot the results
    # Iterate through each npy file in results_folder:
    for filename in os.listdir("results/means"):
        sender_name, receiver_name = filename[:-4].split("_")
        rewards = np.load(f"results/means/{filename}")
        std_dev_rewards = np.load(f"results/std_devs/{filename}")
        rounds = np.arange(len(rewards))
        plt.plot(rounds, np.cumsum(rewards), label=f"{sender_name}, {receiver_name}")
        plt.fill_between(rounds, np.cumsum(rewards) - std_dev_rewards, np.cumsum(rewards) + std_dev_rewards, alpha=0.2)


    title = f"Averaged Cumulative Reward with a {cfg.prior_name} prior over {cfg.n_simulations} simulations with {cfg.n_states} states and {cfg.n_messages} messages"

    plt.xlabel("Round")
    plt.ylabel("Cumulative Reward")
    plt.title(
        line_break(title, 60),
        loc='center'
    )
    plt.legend()
    plt.savefig("plot.png")

if __name__ == "__main__":
    main()