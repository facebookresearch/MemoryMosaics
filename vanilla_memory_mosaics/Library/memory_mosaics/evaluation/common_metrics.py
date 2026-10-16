# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

from collections import OrderedDict
import torch
import numpy as np


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, dataloaders, ctx, eval_iters=10, device="cuda"):
    out = OrderedDict()
    model.eval()
    for key in dataloaders:
        losses = np.zeros(eval_iters)
        eval_status = None 
        for k in range(eval_iters):
            var  = next(dataloaders[key])
            X, Y = var
            storyid = None

            X = (
                X.pin_memory().to(device, non_blocking=True)
                if "cuda" in device
                else X.to(device)
            )
            Y = (
                Y.pin_memory().to(device, non_blocking=True)
                if "cuda" in device
                else Y.to(device)
            )

            with ctx:
                if Y.dim() == 2:
                    _, loss = model(X, Y)
                elif Y.dim() == 3:
                    _, loss = model(X, Y[:,0].contiguous(), Y)
                else:
                    raise NotImplementedError

            losses[k] = loss.item()

            try:
                raw_model = model.module 
            except:
                raw_model = model
 
            try:
                if eval_status is None:
                    eval_status = np.array(raw_model.eval_status)
                else:
                    eval_status += np.array(raw_model.eval_status)
            except:
                pass


        if eval_status is not None:
            eval_status /= eval_iters
            for i, var in enumerate(eval_status):
                out[key + f"_{i}es"] = var 
            
        out[key] = losses.mean()
        out[key + "_squaremean"] = (losses**2).mean()
    
    model.train()
    return out

