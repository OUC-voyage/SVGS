# SV-GS

## 1. Environment

Package versions are in `environment.yml`. Create the environment from the repository root:

```bash
git clone <repository-url>.git --recursive
cd SV-GS
conda env create --file environment.yml
conda activate SVGS
```

Test environment: one NVIDIA GeForce RTX 3090 (24 GB), Intel Xeon Silver 4216 CPU @ 2.10 GHz, Ubuntu 20.04.4 LTS.

## 2. Fixed scene lists

Waymo (`script/run_training_waymo.sh`):

`segment-102751`, `segment-105969`, `segment-106762`, `segment-117240`, `segment-131967`, `segment-140045`, `segment-143481`, `segment-144248`

KITTI-360 (`script/run_training_kitti360.sh`):

`0000`, `0002`, `0003`, `0004`, `0005`, `0006`, `0007`, `0009`, `0010`

PandaSet (`script/run_training_pandaset.sh`):

`028`, `034`, `039`, `040`

## 3. Data and checkpoints

- Dataset: https://drive.google.com/drive/folders/1PGARZQe-bDbsGLTokuLObD09Ja3v-5tD?usp=drive_link
- Lidar Ground True: https://pan.baidu.com/s/1UEdWmJ3SYeFfK0HJsgB86A?pwd=u3e2 提取码: u3e2
- Depth Anything V2 ViT-L and SegFormer-B2 ADE20K: https://drive.google.com/drive/folders/1D0_vS6qzkJZtWwef_gwHzs49LQJzOxFV?usp=drive_link

Expected layout after `mkdir -p data`:

```text
data/
└── Waymo/                      # or KITTI360/ or Pandaset/
    └── <scene>/
        ├── images/             # RGB frames
        ├── sparse/0/           # COLMAP cameras and points
        ├── depths/             # one depth image per RGB frame, same file name
        └── mask/               # one sky mask per frame: <stem>.npy
```

## 4. Preprocessing

### 4.1 Monocular depth (Depth Anything V2, ViT-L)

License: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Checkpoint file: `depth_anything_v2_vitl.pth`.

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
pip install -r requirements.txt
# put depth_anything_v2_vitl.pth in ./checkpoints/
python run.py --encoder vitl --img-path <scene>/images --outdir <scene>/depths --pred-only
```

Copy or rename the outputs so each file in `depths/` has the same file name as the RGB image used by COLMAP.

### 4.2 Sky masks (SegFormer-B2, ADE20K)

License: [NVIDIA Source Code License](https://github.com/NVlabs/SegFormer/blob/master/LICENSE). Checkpoint directory: `models--nvidia--segformer-b2-finetuned-ade-512-512`.

```bash
git clone https://github.com/NVlabs/SegFormer
cd SegFormer
# official example environment: CUDA 10.1, PyTorch 1.7.1
pip install torchvision==0.8.2
pip install timm==0.3.2
pip install mmcv-full==1.2.7
pip install opencv-python==4.5.1.48
pip install -e . --user
python mask_sky.py --model_path <path-to-models--nvidia--segformer-b2-finetuned-ade-512-512> --input_folder <scene>/images
```

Write one `<stem>.npy` per image into `<scene>/mask/`. The loader accepts `.jpg`, `.png`, and `.jpeg` and looks up `mask/<stem>.npy`.

## 5. Training

```bash
bash script/run_training_waymo.sh
bash script/run_training_kitti360.sh
bash script/run_training_pandaset.sh
```

## 6. Rendering and metrics

### 6.1 Novel-view rendering and FPS

```bash
python render.py -s "./data/Waymo/segment-102751" -m "./SVGS/Waymo/segment-102751" --iteration 30000 --skip_train
python metrics.py -m "./SVGS/Waymo/segment-102751"
```

### 6.2 Depth

```bash
python evaluate_depth.py --model_dir <rendered_depth_dir> --gt_dir <gt_depth_dir>
```

### 6.3 Surface normals

```bash
python evaluate_normal.py --pred_dir <rendered_normal_dir> --gt_dir <reference_normal_dir>
```

## 7. Per-scene training cost and rendering speed

### Waymo

| Time/Scene | seg-102751 | seg-105969 | seg-106762 | seg-117240 | seg-131967 | seg-140045 | seg-143481 | seg-144248 | Mean ± Std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Data loading | 1 min 30 s | 1 min 21 s | 1 min 12 s | 1 min 3 s | 0 min 51 s | 0 min 51 s | 0 min 49 s | 0 min 49 s | 1 min 3 s ± 0 min 15 s |
| Runtime | 67 min 31 s | 60 min 43 s | 72 min 44 s | 57 min 17 s | 52 min 0 s | 52 min 41 s | 52 min 13 s | 56 min 4 s | 58 min 54 s ± 7 min 10 s |
| Monocular Depth | 2 min 19 s | 1 min 59 s | 2 min 08 s | 2 min 04 s | 2 min 06 s | 2 min 05 s | 2 min 06 s | 2 min 05 s | 2 min 7 s ± 0 min 5 s |
| Sky mask | 1 min 17 s | 1 min 04 s | 1 min 04 s | 1 min 04 s | 1 min 05 s | 1 min 05 s | 1 min 06 s | 1 min 05 s | 1 min 6 s ± 0 min 4 s |
| Pruning-score | 1 min 8 s | 0 min 46 s | 1 min 16 s | 1 min 56 s | 1 min 27 s | 1 min 18 s | 1 min 13 s | 0 min 57 s | 1 min 15 s ± 0 min 20 s |
| Peak memory (GB) | 13.55 | 13.47 | 17.64 | 17.69 | 11.83 | 17.70 | 13.71 | 16.42 | 15.25 ± 2.21 |
| FPS | 198 | 228 | 196 | 140 | 168 | 194 | 198 | 214 | 192 ± 25.38 |

### KITTI-360

| Time/Scene | 0000 | 0002 | 0003 | 0004 | 0005 | 0006 | 0007 | 0009 | 0010 | Mean ± Std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Data loading | 0 min 29 s | 0 min 32 s | 0 min 27 s | 0 min 30 s | 0 min 27 s | 0 min 32 s | 0 min 22 s | 0 min 31 s | 0 min 28 s | 0 min 29 s ± 0 min 3 s |
| Runtime | 33 min 44 s | 30 min 13 s | 28 min 40 s | 34 min 22 s | 28 min 23 s | 28 min 52 s | 27 min 16 s | 32 min 15 s | 30 min 16 s | 30 min 27 s ± 2 min 21 s |
| Monocular Depth | 2 min 24 s | 2 min 24 s | 1 min 50 s | 1 min 49 s | 1 min 50 s | 2 min 24 s | 1 min 15 s | 2 min 24 s | 2 min 24 s | 2 min 5 s ± 0 min 24 s |
| Sky mask | 0 min 39 s | 0 min 39 s | 0 min 29 s | 0 min 29 s | 0 min 29 s | 0 min 39 s | 0 min 19 s | 0 min 39 s | 0 min 39 s | 0 min 33 s ± 0 min 7 s |
| Pruning-score | 1 min 33 s | 1 min 15 s | 0 min 47 s | 0 min 41 s | 0 min 36 s | 1 min 18 s | 0 min 26 s | 1 min 4 s | 1 min 6 s | 0 min 58 s ± 0 min 21 s |
| Peak memory (GB) | 5.68 | 3.10 | 2.82 | 4.23 | 4.29 | 5.10 | 3.59 | 5.29 | 4.74 | 4.32 ± 0.93 |
| FPS | 177 | 207 | 219 | 216 | 222 | 180 | 236 | 215 | 209 | 209 ± 18.11 |

### PandaSet

| Time/Scene | 028 | 034 | 039 | 040 | Mean ± Std |
| --- | ---: | ---: | ---: | ---: | ---: |
| Data loading | 0 min 30 s | 0 min 37 s | 0 min 33 s | 0 min 34 s | 0 min 34 s ± 0 min 2 s |
| Runtime | 50 min 25 s | 56 min 31 s | 56 min 2 s | 51 min 39 s | 53 min 39 s ± 2 min 40 s |
| Monocular Depth | 0 min 32 s | 0 min 32 s | 0 min 32 s | 0 min 31 s | 0 min 32 s ± 0 min 0 s |
| Sky mask | 0 min 43 s | 0 min 42 s | 0 min 44 s | 0 min 43 s | 0 min 43 s ± 0 min 1 s |
| Pruning-score | 0 min 19 s | 0 min 29 s | 0 min 38 s | 0 min 24 s | 0 min 28 s ± 0 min 7 s |
| Peak memory (GB) | 6.04 | 10.92 | 9.72 | 10.20 | 9.22 ± 1.88 |
| FPS | 246 | 172 | 164 | 210 | 198 ± 32.71 |

## 8. Commit

```bash
git rev-parse HEAD
git status --short
```

## 9. Baseline configuration

Configuration and reproducibility details of the evaluated baseline methods. Repository versions, configurations, and training commands are reported for each method.

| Method | Repository Version | Configuration and Training |
| --- | --- | --- |
| Plenoxels | [ee80e2c](https://github.com/sxyu/svox2/commit/ee80e2c4df8f29a407fda5729a494be94ccf9234) | `opt/configs/custom_alt.json` |
| INGP | [b6e0804](https://github.com/ashawkey/torch-ngp/commit/b6e080468925f0bb44827b4f8f0ed08291dcf8a9) | `scripts/run_nerf.sh` (scene = garden) |
| Mip-NeRF360 | [5b4d4f6](https://github.com/google-research/multinerf/commit/5b4d4f64608ec8077222c52fdf814d40acc10bc1) | `scripts/train_360.sh` (Config.factor = 0) |
| 2DGS | [b13888a](https://github.com/hbb1/2d-gaussian-splatting/commit/b13888adf4d711d259c16710e175fd77d6b333c5) | `scripts/m360_eval.py` (-i images, -r -1) |
| PGSR | [de24f1a](https://github.com/zju3dv/PGSR/commit/de24f1a38b350387e8d8fe381b2cd70c1ae946e7) | `scripts/run_mip360.py` (-r -1) |
| GOF | [5245b20](https://github.com/autonomousvision/gaussian-opacity-fields/commit/5245b20e5d11acd6d1ff5af4b890dc2bedd99693) | `train.py` (-i images, -r -1, --eval) |
| Mini-Splatting | [c0d5581](https://github.com/fatPeter/mini-splatting/commit/c0d55811930dec3bfd3d61cad927093d166aaf82) | `ms/train.py` (-i images, -r -1, --eval, --imp_metric outdoor) |
| Taming3DGS | [fd0f7d9](https://github.com/humansensinglab/taming-3dgs/commit/fd0f7d9edfe135eb4eefd3be82ee56dada7f2a16) | `train.py` (-i images, -r -1, --budget 15, --mode multiplier, --densification_interval 500, --sh_lower) |
| LightGaussian | [6676b98](https://github.com/VITA-Group/LightGaussian/commit/6676b983e77baadd909effc56a6aaadafa964dcc) | `prune_finetune.py`, `distill_train.py`, `vectree/vectree.py` (-i images, -r -1) |
| 3DGS | [54c035f](https://github.com/graphdeco-inria/gaussian-splatting/commit/54c035f7834b564019656c3e3fcc3646292f727d) | `train.py` (-i images, -r -1) |
| AbsGS | [87d7a5a](https://github.com/TY424/AbsGS/commit/87d7a5aac21c0b237a6af9e715e3a0c7df4902d3) | `train.py` (-i images, -r -1) |
| Mip-Splatting | [746a17c](https://github.com/autonomousvision/mip-splatting/commit/746a17c9a906be256ed85b8fe18632f5d53e832d) | `train.py` (-i images, -r -1, --kernel_size 0.1) |
| DashGaussian | [11021fa](https://github.com/YouyuChen0207/DashGaussian/commit/11021fa385a2ec87a7c01fe4b0b7f1c12fb219c3) | `train_dash.py` (-i images, -r -1, --densify_mode freq, --resolution_mode freq, --densify_until_iter 27000) |
| HoGS | [c154481](https://github.com/huntorochi/HoGS/commit/c154481769794d81c8fc5b73177f0fce044024c3) | `train.py` (-i images, -r -1, --eval, --test_iterations -1) |
| Vol3DGS | [52428f7](https://github.com/chinmay0301ucsd/Vol3DGS/commit/52428f7eed7f5c3002c0a83edca4c97cd1359463) | `scripts/all_datasets.sh` (garden) |
| 3D-HGS | [dde1bd5](https://github.com/lihaolin88/3DHGS/commit/dde1bd5e08576dc424ee8a5162ba349251c8ae2d) | `python train.py -s /path/to/your_scene -m /path/to/output --quiet --eval --test_iterations -1` |
| Octree-GS | [9db64fe](https://github.com/city-super/Octree-GS/commit/9db64fe8981f5d6a1002c41cbc275a14f4d5ea0c) | `train_mipnerf360.sh` (visible_threshold=0.01) |
| Pixel-GS | [05c9d55](https://github.com/zhengzhang01/Pixel-GS/commit/05c9d557aa3749bb1ff05e2627943afdd9c18ffd) | `script/mipnerf360.py` (-i images, -r -1) |
| GaussianPro | [df021cd](https://github.com/kcheng1021/GaussianPro/commit/df021cd3bd6d4662694fbfd330bf48deffb52033) | `scripts/waymo.sh` (-i images, -r -1) |
| Scaffold-GS | [59c833b](https://github.com/city-super/Scaffold-GS/commit/59c833b56bbbf510f3f64d40f81721995caced66) | `train_mip360.sh` (garden) |

## License

This project is released under the license in [LICENSE.md](LICENSE.md).
