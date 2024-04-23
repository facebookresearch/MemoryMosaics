# Three Moons

This directory contains code to replicate the three moons experiments
described in section 3 of the paper.  This code relies on the [Apple
MLX framework](https://github.com/ml-explore/mlx) and currently only
runs on Apple Silicon Macs using their integrated GPU, requiring at
least 16GB of RAM (more is better.)  Note that the code says "planets"
whenever the paper says "moons".

First create a virtual python environment with the required packages:
```
$ ./create_venv.sh ~/venv/mlx
$ source ~/venv/mlx/bin/activate
```

Second, run `train_planets-1head.py` to replicate the 1 head
experiments.  Do not forget to close the matplotlib windows to
continue the execution.  Although training is quite fast, the plotting
code is slow because it averages a lot of runs.
```
(mlx) $ python3 train_planets-1head.py
```

Finally, run `train_planets-3heads.py` to replicate the 3 heads experiments.
```
(mlx) $ python3 train_planets-3heads.py
```

