# Memory Mosaics code for In-context Language Learning(ICLL) [benchmark](https://github.com/berlino/seq_icl/tree/main) experiment 


<p align="center">
<image src="figure/dfa_tvd.png" width="700" />
</p>

<p align="center">
<em>
Fig1: Memory Mosaics performance on the RegBench in-context learning benchmark [Akyureket al., 2024]. Since RegBench includes a search over architectural hyper-parameters, Memory Mosaics and transformers use the same search space with the same parameter counts. Memory mosaics outperform all previously tested architectures in this benchmark.
</em>
</p>


This repository provides a minimal code of Memory Mosaics to reproduce the ICLL experiments in Fig1 above.

## Quick Start


### Clone [ICLL](https://github.com/berlino/seq_icl/tree/main) and prepare environments

* Clone ICLL:

```sh
git clone https://github.com/berlino/seq_icl.git
git checkout 9b9223d15348b5a415fb453ed988ed5f7ab9fbdc
```
Now the `seq_icl` folder contains the original code of ICLL benchmark. 

* Then prepare the python environment according to [ICLL readme.md](https://github.com/berlino/seq_icl/blob/main/readme.md). 
```
conda create -n seq_icl python=3.11
pip install -r seq_icl/requirements.txt

```

### Install memory mosaic package
```
cd ../Library
python setup.py install
```

### Plug Memory Mosaics code into ICLL benchmark


* Add `'mm':"src.models.sequence.mm.StackAssoMem",` to `seq_icl/src/utils/registry.py` line 29. 

* Plug `src/models/sequence/mm.py` to `seq_icl/src/models/sequence/`

* Plug `configs/experiment/dfa/mm.yaml` to `seq_icl/configs/experiment/dfa/`


### Evaluate memory Mosaics on ICLL benchmark

```sh
python -m train experiment=dfa/mm
```