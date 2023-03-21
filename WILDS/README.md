# PAIR for WILDS
The dependencies and running commands are the same as for [Fish](https://github.com/YugeTen/fish).
The only difference is that we use `wilds 2.0` following the latest official recommendations.

The results of IRMX and IRMX (PAIR opt) are updated to [Wilds leaderboard](https://wilds.stanford.edu/leaderboard/).
Note there are some slight differences due to the evaluation scripts.
For all methods evaluated in our experiments, we use the default evaluation Fish code provided in Fish and aggregate the results from logs,
which may have minor differences in the variances.
Besides, for CivilComments dataset, we use the first 3 random seeds to tune the hyperparameters following the setting in Fish.
The latest wilds benchmark ask for 5 random seeds, so we directly adopt the hyperparamters tuned via 3 seeds to obtain the results for all seeds.

