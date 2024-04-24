"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train_memory_mosaics.py --batch_size=32 

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train_memory_mosaics.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP xxx.xxx.xxx.xxx and example port xxx:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=xxx.xxx.xxx.xxx --master_port=xxx train_memory_mosaics.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=xxx.xxx.xxx.xxx --master_port=xxx train_memory_mosaics.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import torch
import os
import time
from contextlib import nullcontext
import copy
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import torch.distributed as dist

from utils import create_logger, TimeEstimater
from utils import init_ddp
from utils import str2bool, get_step_lr, get_cosine_lr

import tiktoken
import argparse

# always import numpy after import torch.
import numpy as np

from memory_mosaics.data.dataset import StoriesDataset
from memory_mosaics.data.dataloader import InfiniteDataLoader
from memory_mosaics.evaluation.common_metrics import estimate_loss

#####################
# two version of memory mosaics. Pick one as you wish!
from memory_mosaics.models.memory_mosaics_eft import StackAssoMem
#####################
#from memory_mosaics.models.memory_mosaics import StackAssoMem

parser = argparse.ArgumentParser()

# general 
parser.add_argument('--seed', type=int, default=1337, help='seed ')
parser.add_argument('--out_dir',type=str,default='results/debug', help='output directory')
parser.add_argument('--backend',type=str, default='nccl', help='distributed training backend. [nccl, gloo, etc]. nccl is faster in general.')
parser.add_argument('--device', type=str, default='cuda', help="examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks")
parser.add_argument('--dtype', type=str, default='float16', help="'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler")
parser.add_argument('--compile', type=str2bool, default=True, help=' use PyTorch 2.0 to compile the model to be faster')


# evaluation / logging process
parser.add_argument('--eval_interval', type=int, default=1000, help='validation set evaluation interval')
parser.add_argument('--save_checkpoint_interval', type=int, default=1000, help='save checkpoint (on disk) interval')
parser.add_argument('--log_interval', type=int, default=100, help='log performance interval')
parser.add_argument('--eval_iters',type=int, default=10, help='validation evaluation iterations (per gpu)')

# dataset 
parser.add_argument('--dataset', type=str, default='BabiStories', help='dataset: BabiStories')

# training process 
parser.add_argument('--batch_size',type=int, default=8, help='batch size per gpu')
parser.add_argument('--learning_rate',type=float, default=5e-3, help='learning rate')
parser.add_argument('--max_iters', type=int, default=80000, help='training iterations')
parser.add_argument('--weight_decay',type=float, default=1e-1, help='weight decay')

parser.add_argument('--beta1',type=float, default=0.9, help='beta1 in adam')
parser.add_argument('--beta2',type=float, default=0.95, help="beta2 in adam")
parser.add_argument('--grad_clip',type=float, default=1, help=' clip gradients at this value. or disable if == 0.0')

parser.add_argument('--decay_type',type=str, default='cosine', help='learning rate schaduler')
parser.add_argument('--warmup_iters',type=int, default=2000, help='learning rate warmup iterations')
parser.add_argument('--lr_decay_iters',type=int, default=80000, help='learning rate decay iterations')
parser.add_argument('--min_lr',type=float, default=1e-4, help='min learning rate')

parser.add_argument('--gamma', type=float, default=0.2, help="lr decay factor in steplr")
parser.add_argument('--milestone', type=int, nargs="*", default=[10000], help="milestone in steplr")

# model
parser.add_argument('--block_size',type=int, default=512, help='block size, aka in-context length')
parser.add_argument('--n_layer', type=int, default=12, help='num layers')
parser.add_argument('--n_head', type=int, default=12, help='num heads per layer')
parser.add_argument('--n_embd', type=int, default=768, help='embedding dim')
parser.add_argument('--v_shift',type=int, default=1, help='value right shift')
parser.add_argument('--att_shift',type=int, default=0, help='additional attn shift')


parser.add_argument('--pmem_size', type=int, default=2688, help='memory size')
parser.add_argument('--pmem_count', type=int, default=1, help='memory count')

parser.add_argument('--ic_dropout', type=float, default=0.05, help='in-context attention score dropout rate')
parser.add_argument('--hd_dropout', type=float, default=0.05, help='hidden representation vector dropout rate')

parser.add_argument('--bias', type=str2bool, default=False, help = "do we use bias inside LayerNorm and Linear layers?")
parser.add_argument('--weight_tying', type=str2bool, default=True, help = "True: last linear layer and first embedding share weights. False: do not share.")


parser.add_argument('--pre_ln', type=str2bool, default=True, help="pre-layernorm or post-layernorm. Try post-layernorm if hm_dropout > 0. ")

parser.add_argument('--k_kernel_size',type=int, default=1, help='key kernel size')
parser.add_argument('--v_kernel_size',type=int, default=2, help='value kernel size')
parser.add_argument('--k_fe_type', type=str, default='linearconv', help="key feature extractor type")
parser.add_argument('--v_fe_type', type=str, default='lowrlinearconv', help="value feature extractor type")

parser.add_argument('--skip_tokens', type=int, default=0, help="skip first skip_tokens tokens in loss function. ")
#parser.add_argument('--v_leaky', type=str2bool, default=False, help="value leaky average")
parser.add_argument('--gradient_accumulation_steps',type=int, default=1, help='accumulate gradient to simulate large batch')
parser.add_argument('--attn_only',type=str2bool, default=False, help='only attention blocks.')

args = parser.parse_args()


master_process, seed_offset, ddp_world_size, ddp_rank, device, ddp, rank, ddp_local_rank = init_ddp(args.device, args.backend)

if master_process:
    print('###############parameters##################')
    for key in args.__dict__:
        print(f"{key.ljust(25)} :  {args.__dict__[key]}")
    print('###########################################')

if master_process:
    os.makedirs(args.out_dir, exist_ok=True)

tokens_per_iter = ddp_world_size * args.batch_size * args.block_size
logger = create_logger(os.path.join(args.out_dir, "train.log"), rank=rank)
init_from = "resume" if os.path.isfile(os.path.join(args.out_dir, "ckpt.pt")) else 'scratch'
if master_process:
    logger.info(f"init_from={init_from}")
    logger.info(f"tokens per iteration will be: {tokens_per_iter:,}")

torch.manual_seed(args.seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = { "float32": torch.float32, "bfloat16": torch.bfloat16,"float16": torch.float16, }[args.dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# attempt to derive vocab_size from the dataset
data_dir = os.path.join("memory_mosaics/data", args.dataset)

# ok let's assume gpt-2 encodings by default.
if master_process:
    logger.info(" assuming GPT-2 encodings...")
enc = tiktoken.get_encoding("gpt2")
assert enc.max_token_value == enc.eot_token
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)


if args.dataset in ["BabiStories"]:
    train_dataset = StoriesDataset(data_dir, block_size=args.block_size, split="train")#, future_tokens=args.future_tokens, return_storyid=args.use_storyid)
    val_dataset = StoriesDataset(data_dir, block_size=args.block_size, split="val")#, future_tokens=args.future_tokens, return_storyid=args.use_storyid)
    if master_process:
        logger.info(f'train tokens {len(train_dataset):,}')
else:
    raise NotImplementedError

train_loader = iter(InfiniteDataLoader(train_dataset, batch_size=args.batch_size, num_workers=0))
val_loader = iter(InfiniteDataLoader(val_dataset, batch_size=args.batch_size, num_workers=0))

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
conf = copy.deepcopy(args)
conf.vocab_size = 50304
logger.info("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")

model = StackAssoMem(vocab_size = conf.vocab_size, n_head=conf.n_head, n_embd=conf.n_embd, n_layer=conf.n_layer, \
                    v_shift=conf.v_shift, block_size=conf.block_size, k_fe_type=conf.k_fe_type, v_fe_type=conf.v_fe_type, \
                    k_kernel_size=conf.k_kernel_size, v_kernel_size=conf.v_kernel_size, \
                    ic_dropout=conf.ic_dropout, hd_dropout=conf.hd_dropout, 
                    bias=conf.bias, pmem_count=conf.pmem_count, pmem_size=conf.pmem_size, pre_ln=conf.pre_ln, \
                    weight_tying=conf.weight_tying, skip_tokens=conf.skip_tokens, attn_only=conf.attn_only, \
                    config=conf)

# resume
if init_from == "resume":
    logger.info(f"Resuming training from {args.out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model"]

    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    #model.load_cache(checkpoint["cache"])
    iter_num = checkpoint["iter_num"]

model.to(device)


# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers( args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)
if init_from == "resume":
    if checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

checkpoint = None  # free up memory

# compile the model
if args.compile:
    if master_process:
        logger.info("compiling the model... (takes a ~minute)")
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank],gradient_as_bucket_view=True)


t0 = tstart = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
timeestimater = TimeEstimater(iter_num, args.max_iters)


# check random generator
g = torch.Generator(device=device)
g = g.manual_seed(123)
assert torch.randint(0,999999,size=(1,),generator=g, device=device)[0]==538683 


##################### training    #####################
while True:


    # determine and set the learning rate for this iteration
    if args.decay_type == "cosine":
        lr = get_cosine_lr(iter_num, args.learning_rate, args.min_lr, args.lr_decay_iters, args.warmup_iters)
    else:
        lr = get_step_lr(iter_num, args.learning_rate, args.milestone, args.gamma, args.warmup_iters)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets 
    if iter_num % args.eval_interval == 0 and (iter_num != 0):
        
        losses = estimate_loss(model, {"train": train_loader, "val": val_loader},
            ctx=ctx, eval_iters=args.eval_iters, device=device)
        #raw_model.set_causal(args.causal_rate)

        if ddp:  # distributed evaluation. each GPU estimates {eval_iters} batches. Then average.
            scores = torch.Tensor(np.array([*[losses[key] for key in losses.keys()], 1.0])).to(device)
            dist.all_reduce(scores, op=torch.distributed.ReduceOp.SUM)
            for i, key in enumerate(losses.keys()):
                losses[key] = scores[i].item() / scores[-1].item()

        checkpoint = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_args": conf.__dict__,
            "iter_num": iter_num,
            #"best_val_loss": best_val_loss,
        }
        if master_process:
            logger.info(
                f"[eval] step {iter_num}:"
                + ",".join([f"{key} loss {losses[key]:.4f}" for key in losses]))
            logger.info(f"saving checkpoint to {args.out_dir}")
            torch.save(checkpoint, os.path.join(args.out_dir, "ckpt.pt"))
            
    if iter_num % args.save_checkpoint_interval == 0 and (iter_num !=0):
        checkpoint = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_args": conf.__dict__,
            "iter_num": iter_num,
        }
        if master_process:
            torch.save(checkpoint, os.path.join(args.out_dir, f"ckpt{iter_num}.pt"))

    # get training batches 
    #X, Y, storyid = next(train_loader) if args.use_storyid else [*next(train_loader), None]
    X, Y = next(train_loader)
    X = (X.pin_memory().to(device, non_blocking=True) if "cuda" in device else X.to(device))
    Y = (Y.pin_memory().to(device, non_blocking=True) if "cuda" in device else Y.to(device))   
    #if storyid is not None:
    #   storyid = (storyid.pin_memory().to(device, non_blocking=True) if "cuda" in device else storyid.to(device))

  
    gradient_accumulation_steps = args.gradient_accumulation_steps

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            _, loss = model(X, Y)

            loss = (
                loss / gradient_accumulation_steps
            )  # scale the loss to account for gradient accumulation
        
        
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()


    # clip the gradient
    if args.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    
 
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % args.log_interval == 0 and master_process:
        lossf = loss.item() * args.gradient_accumulation_steps

        VREM = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0 * 1024.0)
        utilization = torch.cuda.utilization()
        logger.info(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms / step, {timeestimater.gettime_format()}, VREM {VREM:.1f} GB, utilization {utilization:.1f}%")


    timeestimater.step(dt * 1000)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > args.max_iters:
        break

if master_process:
    logger.info( f"total time {(time.time() - tstart)/3600:.1f} h, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")




