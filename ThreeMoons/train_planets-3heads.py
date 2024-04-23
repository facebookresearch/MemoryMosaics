# %%
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import matplotlib.pyplot as plt
import tqdm

from planets import PlanetsDataset
from model import ContextMemory
from argparse import Namespace
plt.style.use('bmh')

seed = 100
np.random.seed(seed)
mx.random.seed(seed)
mx.set_default_device(mx.gpu) # mx.gpu/mx.cpu

config = Namespace(
    input_dim=6,
    n_head=3,
    h_dim=2,
    bias=False,
    block_size=800,
    batch_size=64,
    min_periods_in_block_size = 3,
    v_shift=1,
    v_ban=0,
    complex = True,
    norm_keys = False,
    scale_keys= False,
    norm_vals = False,
    attn_scale = 50,
    out_proj=True,
    out_sum=False,
    emb=False,
    eval_batch_size=512, # set eval_batch_size to 1 to speed up plotting
)

model = ContextMemory(config)
mx.eval(model.parameters())

total_iters = 500
scheduler = optim.linear_schedule(1e-2, 5e-6, total_iters)
optimizer = optim.AdamW(learning_rate=scheduler, betas=(.9,.999), weight_decay=0)

train_dataset = PlanetsDataset(block_size=config.block_size, split="train", y_addition=0,
                               min_periods_in_block_size=config.min_periods_in_block_size,
                               observer="sun")

val_dataset = PlanetsDataset(block_size=config.block_size, split="val", y_addition=0,
                             min_periods_in_block_size=config.min_periods_in_block_size,
                             observer="sun")

def batch_iterate(batchsize, dataset):
    while True:
        ix = range(batchsize) # indices do not matter
        xy = [ dataset[i][0:2] for i in ix ]
        yield list(map(mx.stack, zip(*xy)))

train_loader = batch_iterate(config.batch_size, train_dataset)
val_loader = batch_iterate(1, val_dataset)

# plot data
if False:
    x, y, period, periods = train_dataset.get(0)
    plt.plot(*x[:,0:2].T, "b.-", alpha=0.5)
    plt.plot(*x[:,2:4].T, "g.-", alpha=0.5)
    plt.plot(*x[:,4:6].T, "r.-", alpha=0.5)
    plt.show()

# train model
model.train()

def check_nan(x):
    if type(x) == dict:
        for y in x.values():
            check_nan(y)
    elif type(x) == list:
        for y in x:
            check_nan(y)
    elif mx.any(mx.isnan(x)):
        assert False
    
def loss_fn(model, x, y):
    yhat,_,_ = model(x)
    check_nan(yhat)
    #---
    diff = mx.clip(yhat - y, a_min=-.5, a_max=.5)
    return diff.square().mean()
    #----
    #diff = yhat - y
    #diff = diff * mx.clip(diff, a_min=-.25, a_max=+.25)
    #return diff.abs().mean()
    #---
    #return nn.losses.mse_loss(yhat, y, reduction='mean')
    #---
    #return nn.losses.mse_loss(yhat[...,100:,:], y[...,100:,:], reduction='mean')

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

for idx in (bar := tqdm.trange(total_iters)):
    x, y = next(iter(train_loader))
    loss, grads = loss_and_grad_fn(model, x, y)  # what about gradient clipping?
    check_nan(grads)
    optimizer.update(model, grads)
    mx.eval(model.parameters())
    # scheduler.step()
    bar.set_description(f"loss: {loss.item():.4f}")

model.eval()


# %%
# generate and plot once
def plotgen():
    plt.subplots(2, 1, figsize=(9, 6))
    for i, ds in enumerate([train_dataset, val_dataset]):
        plt.subplot(2, 1, i + 1)
        x, y, period, periods = ds.get(5 - 5*i)
        out,*_ = model(x[None,...])
        thresh = config.block_size // 2
        gend = model.generate(x[None, :thresh, :], config.block_size - thresh)
        # crosses and lines (g.-)
        plt.title("train dataset" if i == 0 else "val dataset")
        plt.plot(y.mean(-1), "bx-", label="real", alpha=0.5)
        plt.plot(out[0].mean(-1), "g.-", label="teacherforced", alpha=0.5)
        plt.plot(
            range(thresh, thresh * 2),
            gend[0].mean(-1)[thresh:],
            "r.-",
            label="generated",
            alpha=0.5,
        )
        plt.ylim(-1, 1)
        for p in periods:
            plt.axvline(p, color="grey", alpha=1)
        plt.axvline(period, color="r")
        plt.legend(loc="upper right")
    plt.show()
plotgen()
    
# plot the generation error over n_gen_steps steps vs. context length
def plotctx(n_gen_steps=20):
    np.random.seed(seed)
    fig, _ = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
    
    fig.suptitle(f"generation error over {n_gen_steps} steps vs. context length")
    #fig.supylabel(f"mean(abs(generated - truth)) over {n_gen_steps} steps")
    #fig.supylabel(f"mean(abs(generated - truth))")
    
    for i, ds in enumerate([train_dataset, val_dataset]):
        plt.subplot(1, 2, i + 1)
        plt.ylim(-.05, 1)
        plt.title("train dataset" if i == 0 else "val dataset", fontsize=12)
        #x, y, period, periods = ds.get(5 - 5*i)
        x, y, period, periods = ds.get_batch_with_shifted_startpoint(5 - 5*i, batchsize=config.eval_batch_size)

        diffs = []
        for thresh in tqdm.tqdm(range(1, config.block_size - n_gen_steps)):
            #gend = model.generate(x[:, :thresh], n_gen_steps)[0]
            gend = model.generate(x[:, :thresh], n_gen_steps)
            diff = np.array((y[:, thresh-1:n_gen_steps+thresh-1] - gend[:, thresh:n_gen_steps+thresh]).abs()).mean()
            diffs.append(diff)

        plt.plot(diffs, "b-", label="real - generated", alpha=0.5)
        for p in periods:
            plt.axvline(p-1, color="grey", alpha=1)
            plt.text(p, 0.75, "period", rotation=90, verticalalignment="center")
        plt.axvline(period-1, color="r")
        plt.text(period, 0.75, "total period", rotation=90, verticalalignment="center", color="r")
        #if i == 1:
        #    plt.xlabel("context length for generation")
        if i==0:
            plt.ylabel(f"mean(abs(generated - truth))")
        plt.xlabel("context length for generation")
        

    plt.tight_layout()
    plt.show()
plotctx(25)

# %%
# plot the attentions
def plotatt():
    x, y, period, periods = val_dataset.get(0)
    out,_,att = model(x[None])
    for i in range(model.n_head):
      plt.imshow(np.array(att[0, i][:int(period)+100, :int(period)+100]))
      for p in periods:
        plt.axhline(p-1, color="grey")
      plt.axhline(period-1, color="red")
      plt.show()
plotatt()

# share the cmap
def cmap(w):
    if config.complex:
        w = w.reshape(w.shape[0]//2, 2, w.shape[1]//2, 2) * 0.5
        return w.square().sum(axis=(1,3)).sqrt()
    else:
        return w.abs()
def plotcmap():
    nplots = 3 if config.out_proj else 2
    plt.subplots(1, nplots, figsize=(3 + 3 * nplots, 4))
    plt.subplot(1, nplots, 1)
    plt.imshow(np.array(cmap(model.k_lin.weight)), cmap="viridis")
    plt.colorbar()
    plt.title("$W_{\\varphi}$")
    plt.axis('off')
    plt.subplot(1, nplots, 2)
    plt.imshow(np.array(cmap(model.v_lin.weight)), cmap="viridis")
    plt.colorbar()
    plt.title("$W_{\\psi}$")
    plt.axis('off')
    if config.out_proj:
        plt.subplot(1, nplots, 3)
        plt.imshow(np.array(cmap(model.out_proj.weight)), cmap="viridis")
        plt.colorbar()
        plt.title("$W_{z}$")
        plt.axis('off')
    plt.show()
plotcmap()

