# Memory Mosaics

https://arxiv.org/abs/2405.06394

Memory Mosaics are networks of associative memories working in concert to achieve a prediction task of interest. Like transformers, memory mosaics possess compositional capabilities and in-context learning capabilities. Unlike transformers, memory mosaics achieve these capabilities in comparatively transparent ways. We demonstrate these capabilities on toy examples and we also show that memory mosaics perform as well or better than transformers on medium-scale language modeling tasks.

**License**: This code is released under the [Apache-2.0 License](LICENSE.txt).

This repositories contains several subdirectories relevant to the paper.

* [`BabiStories`](BabiStories) : the BabiStories dataset
* [`ThreeMoons`](ThreeMoons) : The [mlx](https://github.com/ml-explore/mlx) code for the three moon experiments
* [`ICLL`](ICLL): Memory Mosaics code for the In-context Language Learning experiments
* [`Library`](Library): A [pytorch](https://pytorch.org/) library of efficient Memory Mosaics implementation for language tasks. 
* [`nanoMosaics`](nanoMosaics): An alternate [pytorch](https://pytorch.org/) implementation derived fron nanoGPT.
