# STitch3D
[![DOI](https://zenodo.org/badge/567093619.svg)](https://zenodo.org/badge/latestdoi/567093619) [![PyPI](https://img.shields.io/pypi/v/stitch3d?color=green)](https://pypi.python.org/pypi/stitch3d/) [![PyPi license](https://badgen.net/pypi/license/stitch3d/)](https://pypi.org/project/stitch3d/) [![Downloads](https://static.pepy.tech/personalized-badge/stitch3d?period=total&units=international_system&left_color=grey&right_color=orange&left_text=downloads)](https://pepy.tech/project/stitch3d) [![Stars](https://img.shields.io/github/stars/YangLabHKUST/STitch3D?logo=GitHub&color=yellow)](https://github.com/YangLabHKUST/STitch3D/stargazers) 

*Construction of a 3D whole organism spatial atlas by joint modelling of multiple slices with deep neural networks*

An effective and efficient 3D analysis method using spatial transcriptomics data.

Check out our manuscript in Nature Machine Intelligence:
+ [Nature Machine Intelligence website](https://www.nature.com/articles/s42256-023-00734-1)
+ [Preprint in bioRxiv](https://doi.org/10.1101/2023.02.02.526814)

![STitch3D\_pipeline](demos/Overview.jpg)

We developed STitch3D, a deep learning-based method for 3D reconstruction of tissues or whole organisms. Briefly, STitch3D characterizes complex tissue architectures by borrowing information across multiple 2D tissue slices and integrates them with a paired single-cell RNA-sequencing atlas.

With innovations in model designs, STitch3D enables two critical 3D analyses: First, STitch3D detects 3D spatial tissue regions which are related to biological functions, for example cortex layer structures in brain; Second, STitch3D infers 3D spatial distributions of fine-grained cell types in tissues, substantially improving the spatial resolution of seq-based ST approaches. The output of STitch3D can be further used for various downstream tasks like inference of spatial trajectories, denoising of spatial gene expression patterns, identification of genes enriched in specific biologically meaningful regions and detection of cell type gradients in newly generated virtual slices.

An example: STitch3D reconstructed the adult mouse brain, detected 3D layer organizations of the cerebral cortex, and infered curve-shaped distributions of four hippocampal neuron types in three cornu ammonis areas and dentate gyrus.

![hpc](demos/mouse_brain_layers.gif) ![hpc](demos/mouse_brain_hpc.gif)

Installation
------------
* STitch3D can be installed from PyPI:
```bash
pip install stitch3d
```
* Alternatively, STitch3D can be downloaded from GitHub:
```bash
git clone https://github.com/YangLabHKUST/STitch3D.git
cd STitch3D
conda config --set channel_priority strict
conda env update --f environment.yml
conda activate stitch3d
```
Normally the installation time will be less than ten minutes. We have tested our package on Linux (Ubuntu 18.04.5 LTS). Software dependencies are listed in [this website](https://stitch3d-tutorial.readthedocs.io/en/latest/installation.html#software-dependencies).

Tutorials and reproducibility
-----------------------------
We provided codes for reproducing the experiments of the paper "Construction of a 3D whole organism spatial atlas by joint modelling of multiple slices with deep neural networks", and comprehensive tutorials for using STitch3D. Please check the [tutorial website](https://stitch3d-tutorial.readthedocs.io/en/latest/index.html) for more details. 

Interactive 3D results
----------------------
Interactive 3D analysis results from STitch3D are available at the [website](https://stitch3d-tutorial.readthedocs.io/en/latest/index.html).

Reference
----------------------
Gefei Wang, Jia Zhao, Yan Yan, Yang Wang, Angela Ruohao Wu, Can Yang. Construction of a 3D whole organism spatial atlas by joint modelling of multiple slices with deep neural networks. Nature Machine Intelligence (2023).
