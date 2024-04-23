from collections import OrderedDict
import torch
import numpy as np


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, dataloaders, ctx, eval_iters=10, device="cuda"):
    out = OrderedDict()
    model.eval()
    # split_names = ['train','val'] #if data_copy_feq == 0 else ['train','val','valcopy']
    for key in dataloaders:
        losses = np.zeros(eval_iters)
        eval_status = None 
        for k in range(eval_iters):
            var  = next(dataloaders[key])
            # if len(var) == 3:
            #     X, Y, storyid = var 
            # else:
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
            # if storyid is not None:
            #     storyid = (
            #     storyid.pin_memory().to(device, non_blocking=True)
            #     if "cuda" in device
            #     else storyid.to(device)
            #     )

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


# @torch.no_grad()
# def estimate_loss_data(model, X, Y, ctx, eval_iters, storyid=None, device="cuda"):
#     out = OrderedDict()
#     model.eval()

#     assert X.shape[0] % eval_iters == 0
#     batch_size = X.shape[0] // eval_iters

#     losses = np.zeros(eval_iters)
#     for i in range(eval_iters):
#         X_iter = (
#             X[i * batch_size : (i + 1) * batch_size]
#             .pin_memory()
#             .to(device, non_blocking=True)
#             if "cuda" in device
#             else X[i * batch_size : (i + 1) * batch_size].to(device)
#         )
#         Y_iter = (
#             Y[i * batch_size : (i + 1) * batch_size]
#             .pin_memory()
#             .to(device, non_blocking=True)
#             if "cuda" in device
#             else Y[i * batch_size : (i + 1) * batch_size].to(device)
#         )
#         with ctx:
#             logits, loss = model(X_iter, Y_iter, storyid=storyid)

#         losses[i] = loss.item()
#     out["loss"] = losses.mean()
#     out["loss" + "_squaremean"] = (losses**2).mean()
#     model.train()
#     return out


# @torch.no_grad()
# def inference(model, save_inference_memory, X, Y, ctx, batch_size=None, device="cuda"):
#     model.eval()
#     B = X.shape[0]
#     on_cuda = "cuda" in device
#     assert B % batch_size == 0
#     if save_inference_memory:  # turn on when you have less GPU and larger history_size
#         for i in range(B // batch_size):
#             X_iter = (
#                 X[i * batch_size : (i + 1) * batch_size]
#                 .pin_memory()
#                 .to(device, non_blocking=True)
#                 if on_cuda
#                 else X[i * batch_size : (i + 1) * batch_size].to(device)
#             )
#             Y_iter = (
#                 Y[i * batch_size : (i + 1) * batch_size]
#                 .pin_memory()
#                 .to(device, non_blocking=True)
#                 if on_cuda
#                 else Y[i * batch_size : (i + 1) * batch_size].to(device)
#             )
#             with ctx:
#                 model(X_iter, Y_iter)
#     else:
#         X = (
#             X.pin_memory().to(device, non_blocking=True)
#             if on_cuda
#             else X.to(device)
#         )
#         Y = (
#             Y.pin_memory().to(device, non_blocking=True)
#             if on_cuda
#             else Y.to(device)
#         )
#         with ctx:
#             model(X, Y)
#     model.train()
