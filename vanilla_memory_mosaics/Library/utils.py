# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

# This file is derived frrom
# https://github.com/facebookresearch/swav/blob/main/src/logger.py

import os
import logging
import time
from datetime import timedelta
import pandas as pd
import torch
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import signal
from contextlib import contextmanager
import math
import sys
import numpy as np 
import subprocess

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ""


def create_logger(filepath, rank):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = "%s-%i" % (filepath, rank)
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        #file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    #logger.setLevel(logging.INFO)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    return logger

# learning rate decay scheduler (cosine with warmup)
def get_cosine_lr(it, learning_rate, min_lr, lr_decay_iters, warmup_iters):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def get_step_lr(it, learning_rate, milestone, gamma,warmup_iters):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    current_lr = learning_rate
    for var in milestone:
        if it >= var:
            current_lr *= gamma
    return current_lr


class TimeEstimater(object):
    def __init__(self, start_iter, max_iter):
        self.start_iter = start_iter
        self.max_iter = max_iter
        self.iters = 0
        self.timecost = 0

    def step(self, timecost, num_iter=1):
        self.timecost += timecost
        self.iters += num_iter

    def gettime(self):
        per_iter_time = 0 if self.iters == 0 else self.timecost / self.iters
        total_time = self.max_iter * per_iter_time
        remain_time = (self.max_iter - self.start_iter - self.iters) * per_iter_time

        return total_time, remain_time

    def gettime_format(self):
        total_time, remain_time = self.gettime()
        string = "total time %.1f h, remain time %.1f h" % (
            total_time / 1000 / 3600,
            remain_time / 1000 / 3600,
        )
        return string


class PD_Stats(object):

    """
    Log stuff with pandas library
    """

    def __init__(self, path, columns):
        self.path = path

        # reload path stats
        if os.path.isfile(self.path):
            self.stats = pd.read_pickle(self.path)

            # check that columns are the same
            assert list(self.stats.columns) == list(columns)

        else:
            self.stats = pd.DataFrame(columns=columns)

    def update(self, row, save=True):
        self.stats.loc[len(self.stats.index)] = row

        # save the statistics
        if save:
            self.stats.to_pickle(self.path)


class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


# try:
#     with time_limit(10):
#         long_function_call()
# except TimeoutException as e:
#     print("Timed out!")

def communication_test(device):
    k=100
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(k):
        # perform a dummy all-reduce to initialize the NCCL communicator
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).to(device))
    torch.cuda.synchronize()
    t1 = time.time()

    return (t1 - t0) 

def computation_test(device):
    k=5
    
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(k):
        A = torch.normal(0.,1.,size=(10000,10000)).to(device)
        B = torch.normal(0.,1.,size=(10000,10000)).to(device)
        C = A @ B 
    torch.cuda.synchronize()
    t1 = time.time() 
    return (t1 - t0) 



def init_ddp(device, backend="nccl", logger=None):

    # init logger, distributed environment, and random seed
    try:
        rank = int(os.environ.get("RANK"))
    except:
        rank = 0

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    print(f'ddp run {ddp}')
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])

        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (
            ddp_rank == 0
        )  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # assert gradient_accumulation_steps % torch.cuda.device_count() == 0

        dist.barrier()
        
        try: #scontrol show hostnames $SLURM_JOB_NODELIST
            nodelists = str(os.environ["SLURM_JOB_NODELIST"])
            ddp_nodelist = subprocess.check_output(f"scontrol show hostnames {nodelists}".split()).decode('utf-8').strip().split('\n')
            if master_process:
                print(ddp_nodelist)
            nodename = ddp_nodelist[ddp_rank // len(ddp_nodelist)]
            
        except:
            ddp_nodelist = None 
            nodename = f'unknown{ddp_rank}'
        
        try: 
            jobid = str(os.environ['SLURM_JOB_ID'])
        except:
            jobid = str(np.random.randint(0, 9999))



        # try:
        #     with time_limit(5):
        #         comm_t = communication_test(device)
        # except TimeoutException as e:    
        #     raise TimeoutException(f'node {nodename} rank {ddp_rank} localrank {ddp_local_rank} communication Timed out!')
        
        # try:
        #     with time_limit(100):
        #         compute_t = computation_test(device)
        #     status = torch.zeros(1).to(device)
        #     dist.all_reduce(status, op=dist.ReduceOp.SUM)
        #     if status > 0:
        #         raise TimeoutException('certain computation node time out')
        # except TimeoutException as e:
        #     print(e)
        #     print(f'node {nodename} rank {ddp_rank} localrank {ddp_local_rank} computation Seriously Timed out!')
        #     compute_t = 999999

        # if compute_t > 30:
        #     print(f'node {nodename} rank {ddp_rank} localrank {ddp_local_rank} computation timed out {compute_t}!')
            

        # os.makedirs('.slurm_nodes_benckmark/', exist_ok=True)
        # with open(f'.slurm_nodes_benckmark/{nodename}_{ddp_local_rank}.txt','w') as f:
        #     f.write(f'{compute_t:.3f}')
        # f.close()

        # compute_t = torch.ones(1, dtype=torch.float32).to(device) * compute_t
        # compute_t_all = torch.zeros(ddp_world_size, dtype = torch.float32).to(device)
        # dist.all_gather_into_tensor(compute_t_all, compute_t)

        # if compute_t_all.max() > 20:
        #     sys.exit(-1)



        
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_rank = 0
        device = device
        ddp_local_rank = 0
    return (
        master_process,
        seed_offset,
        ddp_world_size,
        ddp_rank,
        device,
        ddp,
        rank,
        ddp_local_rank,
    )

def print_args(args):
    for key in args.__dict__:
        print(f"{key}:{args.key}")
