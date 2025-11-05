# SVRS

![visualization2](OverAll.png)

##  🌼 Abstract
Three-dimensional voxel reconstruction based on stereo vision is essential for environmental perception in autonomous robots. Existing pseudo-LiDAR methods recover voxel grids by estimating depth maps and projecting them pixel by pixel, leading to high computational cost and boundary over-smoothing. To overcome these issues, we model the inverse relationship between 2D pixels and 3D voxel grids and propose a Self-supervised 3D Voxel Reconstruction network from Stereo vision (SVRS). Specifically, we represent a given 3D scene as multi-scale uniform cubic voxel grids and introduce. PVPM projects the 3D position of each voxel grids into index coordinates, which establishes implicit stereo–voxel correspondences and converts dense pixel features into sparse voxel representations. Furthermore, we explore an Octree-based Encoder-Decoder Architecture (OEDA) to reconstruct multi-scale voxel grids via hierarchical spatial partitioning, avoiding the influence of dense empty grids on sparse occupied grids from coarse-to-fine. Finally, SVRS leverages off-the-shelf stereo matching methods within a self-supervised training framework. Experiments on the DrivingStereo dataset show that SVRS achieves competitive reconstruction accuracy while improving inference speed by up to 14× over advanced pseudo-LiDAR approaches and 3× over real-time stereo methods.

## :art: Zero-shot performance in ill-posed regions
![visualization2](Zero-shot.png)
Comparison in ill-posed regions.

## ⚙️ Installation
* NVIDIA RTX 4090
* python 3.8

### ⏳ Create a virtual environment and activate it.

```Shell
conda create -n monster python=3.8
conda activate monster
```
### 🎬 Dependencies

```Shell
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install tqdm
pip install scipy
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install timm==0.6.13
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install accelerate==1.0.1
pip install gradio_imageslider
pip install gradio==4.29.0

```

# Acknowledgements

This project is based on [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), [IGEV](https://github.com/gangweiX/IGEV), [Stereo Anywhere](https://github.com/bartn8/stereoanywhere), and [Monster](https://github.com/Junda24/MonSter).
