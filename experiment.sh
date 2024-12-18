#!/bin/bash
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem=1GB
#SBATCH --job-name=bandit
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/errors/%x_%j.err

directory=outputs/$(date '+%Y-%m-%d_%H-%M-%S')

n_simulations=100

n_rounds=10000
n_states=10
n_messages=3

utility_name=equality
prior_name=skewed

seed=$RANDOM

source .venv/bin/activate

for threshold in 0.001
do
    python3 main.py \
        hydra.run.dir=$directory \
        hydra.job.chdir=True \
        seed=$seed \
        n_rounds=$n_rounds \
        n_states=$n_states \
        n_messages=$n_messages \
        utility_name=$utility_name \
        prior_name=$prior_name \
        sender_name=threshold \
        threshold=$threshold \
        receiver_name=exp3 \
        eta=1

    python3 main.py \
        hydra.run.dir=$directory \
        hydra.job.chdir=True \
        seed=$seed \
        n_rounds=$n_rounds \
        n_states=$n_states \
        n_messages=$n_messages \
        utility_name=$utility_name \
        prior_name=$prior_name \
        sender_name=threshold \
        threshold=$threshold \
        receiver_name=ucb \
        delta_exponent=2
    
    python3 main.py \
        hydra.run.dir=$directory \
        hydra.job.chdir=True \
        seed=$seed \
        n_rounds=$n_rounds \
        n_states=$n_states \
        n_messages=$n_messages \
        utility_name=$utility_name \
        prior_name=$prior_name \
        sender_name=threshold \
        threshold=$threshold \
        receiver_name=etc
    
    python3 main.py \
        hydra.run.dir=$directory \
        hydra.job.chdir=True \
        seed=$seed \
        n_rounds=$n_rounds \
        n_states=$n_states \
        n_messages=$n_messages \
        utility_name=$utility_name \
        prior_name=$prior_name \
        sender_name=threshold \
        threshold=$threshold \
        receiver_name=greedy \
        epsilon=0.01
done

python3 plot.py \
    hydra.run.dir=$directory \
    hydra.job.chdir=True \
    n_simulations=$n_simulations \
    n_messages=$n_messages \
    n_states=$n_states \
    prior_name=$prior_name