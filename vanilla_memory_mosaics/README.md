# Memory Mosaics


Memory Mosaics are networks of associative memories working in concert to achieve a prediction task of interest. Like transformers, memory mosaics possess compositional capabilities and in-context learning capabilities. Unlike transformers, memory mosaics achieve these capabilities in comparatively transparent ways. We demonstrate these capabilities on toy examples and we also show that memory mosaics perform as well or better than transformers on medium-scale language modeling tasks.

**Latest News**
- [2025.09] [Memory Mosaics at scale](https://proceedings.neurips.cc/paper_files/paper/2025/file/3767842e2ebee900236f534855aa2c36-Paper-Conference.pdf) scales Memory Mosaics to 10B parameters and 1T training tokens, accepted by NeurIPS 2025 **Oral (top 0.36%)**.
- [2025.01] [Memory Mosaics](https://proceedings.iclr.cc/paper_files/paper/2025/file/59c3ac496e6fe7be0c2c2b95014e2210-Paper-Conference.pdf) is accepted by ICLR 2025.



**License**: This code is released under the [Apache-2.0 License](LICENSE.txt).



The following folders contain dataset, code, and instructions for paper [Memory Mosaics](https://proceedings.iclr.cc/paper_files/paper/2025/file/59c3ac496e6fe7be0c2c2b95014e2210-Paper-Conference.pdf).

* [`BabiStories`](BabiStories) : the BabiStories dataset
* [`ThreeMoons`](ThreeMoons) : The [mlx](https://github.com/ml-explore/mlx) code for the three moon experiments
* [`ICLL`](ICLL): Memory Mosaics code for the In-context Language Learning experiments
* [`Library`](Library): A [pytorch](https://pytorch.org/) library of efficient Memory Mosaics implementation for language tasks. 
* [`nanoMosaics`](nanoMosaics): An alternate [pytorch](https://pytorch.org/) implementation derived from nanoGPT.



## Reference

```
@inproceedings{zhang2025memory,
  title={Memory mosaics},
  author={Zhang, Jianyu and Nolte, Niklas and Sadhukhan, Ranajoy and Chen, Beidi and Bottou, L{\'e}on},
  booktitle={International Conference on Learning Representations},
  volume={2025},
  pages={36412--36433},
  year={2025}
}

@article{zhang2026memory,
  title={Memory Mosaics at scale},
  author={Zhang, Jianyu and Bottou, L{\'e}on},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={38929--38956},
  year={2026}
}
```
