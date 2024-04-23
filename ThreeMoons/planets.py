# %%
import math
import numpy as np
import mlx.core as mx
#import mlx.nn as nn
#import mlx.optimizers as optim
import matplotlib.pyplot as plt
from itertools import combinations
from functools import reduce

def lcm_float(a, b, precision):
    a = round(a, precision)
    b = round(b, precision)
    return math.lcm(int(a * 10**precision), int(b * 10**precision)) / 10**precision


# use itertools reduce to find the lcm of a list of numbers
def lcm_reduce(num_list, precision):
    return reduce(lambda x, y: lcm_float(x, y, precision), num_list)

# %%
class PlanetsDataset:
    # randomly samply chunks as example
    def __init__(self,
                 block_size=512,
                 y_addition=0,
                 split="train",
                 approx_total_tokens=int(1e4),
                 min_periods_in_block_size = 2,
                 observer="sun" ):
        
        # split \in [train, val]
        self.all_periods = np.array(
            list(combinations([0.5, 1, 1.5, 2, 2.5, 3, 3.5], 3))
        )
        self.full_periods = np.array([lcm_reduce(p, 1) for p in self.all_periods])
        # select only bottom 50% of periods with argsort

        selection = np.argsort(self.full_periods)[: len(self.full_periods) // 4 + 1]
        self.all_periods = self.all_periods[selection].tolist()
        self.full_periods = self.full_periods[selection].tolist()
        self.total_period = lcm_reduce(self.full_periods, 1)
        self.max_period = max(self.full_periods)
        self.observer = observer

        self.val_period = self.all_periods[-1]
        self.all_periods.remove(self.val_period)
        self.y_addition = y_addition
        self.train_period = self.all_periods
        self.val_period = [self.val_period]
        self.block_size = block_size

        # define the linspace
        point_spacing_multiple = 4  # assumes periods are half integers
        point_spacing_upper_bound = (block_size / self.max_period / min_periods_in_block_size )
        assert point_spacing_upper_bound >= point_spacing_multiple
        point_spacing = np.arange(0, point_spacing_upper_bound + 1, point_spacing_multiple)[-1]

        # use this if you want exact multiples
        # self.linspace = np.arange(0, approx_total_tokens) / point_spacing
        # use this if not
        self.linspace = np.arange(0, approx_total_tokens) / point_spacing_upper_bound
        # total tokens was only an approximate
        self.total_tokens = len(self.linspace)
        self.shift = 1
        if split == "train":
            self.all_periods = self.train_period
        else:
            self.all_periods = self.val_period
        self.vocab_size = 1

    def __len__(self):
        return 100000  # fake length

    def get(self, idx):
        periods = self.all_periods[idx]
        start_point = np.random.randint(0, self.total_tokens - self.y_addition - self.block_size * 2 - self.shift)
        offset = 0.3
        linspace = self.linspace[start_point : start_point + self.block_size + self.shift + self.y_addition ]
        sines = np.array([np.sin(2 * np.pi / p * linspace + offset) for p in periods])
        cosines = np.array([np.cos(2 * np.pi / p * linspace + offset) for p in periods])
        if self.observer != "sun":
          sines[:-1] -= sines[-1]
          cosines[:-1] -= cosines[-1]
        data = np.stack((sines, cosines), axis=1).reshape(6, -1).T
        x = mx.array(data[: self.block_size].astype(np.float32))
        y = mx.array(data[self.shift : self.block_size + self.shift + self.y_addition].astype(np.float32))
        period = lcm_reduce(periods, 1) / (self.linspace[1] - self.linspace[0])
        periods = [p / (self.linspace[1] - self.linspace[0]) for p in periods]
        return x, y, period, periods

    def __getitem__(self, _):
        return self.get(np.random.choice(len(self.all_periods), 1)[0])


if __name__ == "__main__":
    block_size = 256
    ds = PlanetsDataset(block_size, split="train", observer="planet")
    periods = ds.full_periods
    full_period = lcm_reduce(periods, 1)
    assert full_period == int(full_period)
    for idx in range(len(ds.all_periods)):
        x, y, period, _ = ds[idx]
        plt.figure(figsize=(2, 2))
        plt.scatter(*x[:,:2].T, alpha=.5)
        plt.scatter(*x[:,2:4].T + 0.1, alpha=.5)
        plt.scatter(*x[:,4:].T + 0.2, alpha=.5)
    print(ds.full_periods)

# %%
