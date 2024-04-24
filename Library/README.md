# Memory Mosaics 

A package of Memory Mosaics with a CUDA kernel of the leakyaverage operator. 

## Setup Requirements


```bash
conda create -n mm python=3.8
conda activate mm 
pip install -r requirements.txt
pip install pynvml
```


## Quick start

### Install Memory Mosaics package

```sh 
python setup.py install 
```


This packages provide two versions of memory mosaics implementation: `memory_mosaics.models.memory_mosaics_eft` and `memory_mosaics.models.memory_mosaics`. The first one contains the cuda kernel of leakaverage, and thus fast on long context. The later one use the naive implementation, so that it is easy to modify. 


One can import memory mosaics by  `try ... except ...`:
```
try:
	from memory_mosaics.models.memory_mosaics_eft import StackAssoMem
except:
	from memory_mosaics.models.memory_mosaics import StackAssoMem
```

### Prepare datasets

* Put BabiStories dataset (`traindataset.txt` and `testdataset.txt`) to `memory_mosaics/data/BabiStories`. For example,

```sh
cp ../BabiStories/data/*dataset.txt memory_mosaics/data/BabiStories/
```

* Tokenize the dataset and put them into one larger stream of integers (by numpy.memmap). 

```sh
cd memory_mosaics/data/BabiStories/
python prepare.py
```

* Train Memory Mosaics on BabiStories dataset 

```
python train_memory_mosaics.py --batch_size [batch_size]
```




## Benchmark  

### Speed and Memory 

Distributed training on 2 (Quadro GV100) GPUS (same node). Network architecture 12 layers, 12 heads, and 768 embedding dim. 

```
torchrun --standalone --nproc_per_node=2 train_memory_mosaics.py --compile [True/False] --v_fe_type linearconv
```

|block size|batch size |version|compile| time/step (ms)| VREM (GB) | 
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|512 |8|eft|x| 245  | 7.7 |
|512 |8|eft|√| 189  | 7.7 |
|512 |8|regular|x| 310  | 10.2|
|512 |8|regular|√| 190  | 9.4 |
|2048|2| eft| √ | 247 | 9.0 | 
|2048|2| regular| √ | 301 | 17.4|


### Training and validation loss 

|block size |version|train loss| val loss| 
|:-------:|:-------:|:-------:|:-------:|
|512 |eft| 1.3294  | 1.5008 |
|512 |regular| 1.3337  | 1.4981 |

Considering the randomness during training and evaluation, both versions achieve the same loss. 


### Speed and Memory on larger MM

```
torchrun --standalone --nproc_per_node=2 train_memory_mosaics.py --compile [True/False] --v_fe_type lowrlinearconv
```

|block size|batch size |#params |#layer|#head| #dim| version|compile| time/step (ms)| VREM (GB) | 
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|1024| 1| 354M | 24 |16| 1024|eft|√| 254|9.1|
|1024| 1| 756M | 24 |16| 1536|eft|√| 439|16.2|
|2048| 1| 354M | 24 |16| 1024|eft|√| 415|12.9|


