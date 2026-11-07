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
- Trained SV-GS scenes (large PLY files): https://drive.google.com/drive/folders/1AFzbn_fOO2jjPvzTwGQlHIjI7tDyQqOc?usp=drive_link
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

From the repository root, with `SVGS` activated:

```bash
bash script/run_training_waymo.sh
bash script/run_training_kitti360.sh
bash script/run_training_pandaset.sh
```

## 6. Rendering and metrics

### 6.1 Novel-view rendering and FPS

FPS is printed by `render.py` as `Test FPS` at the 1600 px width policy on the same GPU as training.

```bash
python render.py -s "./data/Waymo/segment-102751" -m "./SVGS/Waymo/segment-102751" --iteration 30000 --skip_train
python metrics.py -m "./SVGS/Waymo/segment-102751"
```

`metrics.py` reports PSNR, SSIM, and LPIPS on `test/ours_30000/renders` against `test/ours_30000/gt`, and writes `results.json`.

Repeat with the KITTI-360 and PandaSet scene paths from Section 2.

### 6.2 Depth

`evaluate_depth.py` compares rendered `.npy` depth with GT depth on valid pixels (`GT > 0`).

```bash
python evaluate_depth.py --model_dir <rendered_depth_dir> --gt_dir <gt_depth_dir>
```

### 6.3 Surface normals

`evaluate_normal.py` reports mean angular error in degrees between rendered normals and the reference `.npy` normals.

```bash
python evaluate_normal.py --pred_dir <rendered_normal_dir> --gt_dir <reference_normal_dir>
```

## License

This project is released under the license in [LICENSE.md](LICENSE.md).

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
