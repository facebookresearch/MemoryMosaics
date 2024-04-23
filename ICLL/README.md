# Memory Mosaics code for In-context Language Learning(ICLL) [benchmark](https://github.com/berlino/seq_icl/tree/main) experiment 

This repository provides a minimal code of Memory Mosaics to plug in the [ICLL benchmark](https://github.com/berlino/seq_icl/tree/main). 


### Clone ICLL and prepare environments

* Clone ICLL:

```sh
git clone https://github.com/berlino/seq_icl.git
git checkout 9b9223d15348b5a415fb453ed988ed5f7ab9fbdc
```

* Then prepare the python environment according to [ICLL](https://github.com/berlino/seq_icl/blob/main/readme.md). 

### Install memory mosaic package

Check the [Library](../Library) folder for details.

### Plug in!


* Add `'mm':"src.models.sequence.mm.StackAssoMem",` to [ICLL](https://github.com/berlino/seq_icl/blob/main/readme.md)`/src/utils/registry.py` line 29. 

* Plug `src/models/sequence/mm.py` to [ICLL](https://github.com/berlino/seq_icl/blob/main/readme.md)`/src/models/sequence/`

* Plug `configs/experiment/dfa/mm.yaml` to [ICLL](https://github.com/berlino/seq_icl/blob/main/readme.md)`/configs/experiment/dfa/`
