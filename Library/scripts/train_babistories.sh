# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

#!/bin/bash
#SBATCH --requeue
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --constraint=volta32gb
#SBATCH --job-name=lowr_v
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:8
#SBATCH --mem=64G
#SBATCH --array=0



nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
export OMP_NUM_THREADS=1
export MASTER_ADDR=$head_node_ip


traindataset=BabiStories
i=0;
for kernel in 2;
do 
	for dropout in 0.05; 
	do 
		for iter in 80000;
		do 
			for n_head in 12; 
			do
				for n_embd in 768;
				do 
					for n_layer in 12; #12;# 18;
					do 
						for learning_rate in 5e-3; 
						do 
							for	weight_decay in 1e-1; 
							do 
								n_heads[$i]=$n_head;
								n_embds[$i]=$n_embd;
								n_layers[$i]=$n_layer;
								learning_rates[$i]=$learning_rate;
								weight_decays[$i]=$weight_decay;
								pmem_sizes[$i]=2688;
								pmem_counts[$i]=1;
								kernels[$i]=$kernel;
								iters[$i]=$iter;
								dropouts[$i]=$dropout;
								i=$(($i+1));
							done 
						done 
					done 
				done 
			done 
		done 
	done 
done

n_head=${n_heads[$SLURM_ARRAY_TASK_ID]}
n_embd=${n_embds[$SLURM_ARRAY_TASK_ID]}
n_layer=${n_layers[$SLURM_ARRAY_TASK_ID]}
learning_rate=${learning_rates[$SLURM_ARRAY_TASK_ID]}
weight_decay=${weight_decays[$SLURM_ARRAY_TASK_ID]}
iter=${iters[$SLURM_ARRAY_TASK_ID]}
dropout=${dropouts[$SLURM_ARRAY_TASK_ID]}
pmem_count=${pmem_counts[$SLURM_ARRAY_TASK_ID]}
pmem_size=${pmem_sizes[$SLURM_ARRAY_TASK_ID]}
kernel=${kernels[$SLURM_ARRAY_TASK_ID]}

# --batch_size indicates "batch size per GPU"! the actual batch size is 8x8x8 in this case. 

EXPERIMENT_PATH=results/${traindataset}/eftversion/count${pmem_count}_size${pmem_size}_nlayer${n_layer}_nembed${n_embd}_nhead${n_head}_lr${learning_rate}_wd${weight_decay}_dp${dropout}_epoch${iter}
srun  --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err  torchrun \
--nnodes 8 \
--nproc_per_node 8 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:13212 \
train_memory_mosaics.py --compile=True --dataset=${traindataset}   \
--v_fe_type=lowrlinearconv --k_fe_type=linearconv --k_kernel_size=1 --v_kernel_size=2 \
--out_dir ${EXPERIMENT_PATH} --weight_tying=True \
--hd_dropout=${dropout} --ic_dropout=${dropout} --att_shift=0 \
--eval_interval=2000 --warmup_iters=2000 --save_checkpoint_interval=10000 \
--n_layer=${n_layer} --n_embd=${n_embd} --n_head=${n_head} \
--learning_rate=${learning_rate} \
--v_shift 1 --pmem_size ${pmem_size} --pmem_count ${pmem_count} \
--weight_decay=${weight_decay} --block_size=512 \
--pre_ln=True --batch_size=8 --max_iters=${iter} --lr_decay_iters=${iter}  --min_lr=1e-4  || scontrol requeue $SLURM_JOB_ID
