import os
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--gt_dir", required=True, type=str)
parser.add_argument("--pred_dir", required=True, type=str)
args = parser.parse_args()

gt_dir = args.gt_dir
pred_dir = args.pred_dir

files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".npy")])

all_mae = []

for f in tqdm(files):
    gt = np.load(os.path.join(gt_dir, f)).astype(np.float32)
    pred = np.load(os.path.join(pred_dir, f)).astype(np.float32)

    # 统一为 (H, W, 3)
    if gt.ndim == 3 and gt.shape[0] == 3:
        gt = np.transpose(gt, (1, 2, 0))

    if pred.ndim == 3 and pred.shape[0] == 3:
        pred = np.transpose(pred, (1, 2, 0))

    # 归一化
    gt = gt / np.clip(np.linalg.norm(gt, axis=-1, keepdims=True), 1e-8, None)
    pred = pred / np.clip(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-8, None)

    # 有效GT
    valid = np.linalg.norm(gt, axis=-1) > 1e-6

    # 无符号Normal MAE（推荐用于PCA法线）
    dot = np.abs(np.sum(gt * pred, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)

    angle = np.degrees(np.arccos(dot))

    all_mae.append(angle[valid].mean())

print("=" * 50)
print("Images:", len(all_mae))
print("Average Normal MAE: %.4f°" % np.mean(all_mae))
print("=" * 50)