# Learning to Converse

Run the experiment.sh file

## Config 

UCB uses the estimate at time step $t$ of $n$ for arm $i$ 

$$
UCB(i, t) = \mu_{t, i} + \sqrt{\text{(delta exponent)}\cdot \frac{\log(n)}{N_i(t)}}
$$

where delta exponent is set in the config.

Exp3 draws from distribution $p$, when there are $k$ arms and $n$ total steps

$$
p \propto \exp\left(\text{eta} \cdot \sqrt{\frac{\log(k)}{nk}}\right)
$$

where eta is set in the config.
