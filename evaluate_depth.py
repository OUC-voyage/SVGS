import os
import numpy as np


def normalize_like_vis(depth):
    """
    与深度可视化相同：用自身 1% / 99% 分位作上下界，裁剪并映射到 [0, 1]。
    上下界退化时返回 None。
    """
    vmin = np.percentile(depth, 1)
    vmax = np.percentile(depth, 99)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return None
    return np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0)


def evaluate_pair(model_path, gt_path):
    """
    对单张图进行深度评估
    GT: LiDAR绝对深度
    Prediction: 相对深度，需要median scaling
    """

    # =========================
    # 加载GT LiDAR深度
    # =========================
    gt_depth = np.load(gt_path).astype(np.float32)

    # =========================
    # 加载预测深度
    # =========================
    pred_depth = np.load(model_path).astype(np.float32)


    # 检查尺寸
    if gt_depth.shape != pred_depth.shape:
        raise ValueError(
            f"Shape mismatch: GT {gt_depth.shape}, Pred {pred_depth.shape}"
        )


    # =========================
    # 只在LiDAR有效区域评价
    # GT > 0 表示有效深度
    # =========================
    valid_mask = gt_depth > 0


    gt = gt_depth[valid_mask]
    pred = pred_depth[valid_mask]


    if len(gt) == 0:
        return None


    # 防止异常值
    valid_pred = pred > 0

    gt = gt[valid_pred]
    pred = pred[valid_pred]


    if len(gt) == 0:
        return None


    # =========================
    # Median Scaling
    # 相对深度 -> 绝对尺度
    # =========================
    scale = np.median(gt) / np.median(pred)

    pred_aligned = pred * scale


    # =========================
    # Abs Relative Error
    # =========================
    abs_rel = np.mean(
        np.abs(pred_aligned - gt) / gt
    )


    # =========================
    # MAE
    # =========================
    mae = np.mean(
        np.abs(pred_aligned - gt)
    )


    # =========================
    # delta < 1.25
    # =========================
    thresh = np.maximum(
        pred_aligned / gt,
        gt / pred_aligned
    )

    d1 = np.mean(
        thresh < 1.25
    )


    # =========================
    # 各自归一化到 [0, 1] 后的平均绝对误差
    # =========================
    gt_n = normalize_like_vis(gt)
    pred_n = normalize_like_vis(pred)
    if gt_n is None or pred_n is None:
        norm_mae = np.nan
    else:
        norm_mae = np.mean(np.abs(pred_n - gt_n))


    return abs_rel, mae, d1, norm_mae



def evaluate_all(model_dir, gt_dir):
    """
    批量评估所有深度文件
    """

    abs_rels = []
    maes = []
    d1s = []
    norm_maes = []


    file_list = sorted([
        f for f in os.listdir(gt_dir)
        if f.endswith(".npy")
    ])


    for filename in file_list:

        gt_path = os.path.join(
            gt_dir,
            filename
        )

        pred_path = os.path.join(
            model_dir,
            filename
        )


        if not os.path.exists(pred_path):
            print(
                f"[跳过] 缺少预测深度: {filename}"
            )
            continue


        result = evaluate_pair(
            pred_path,
            gt_path
        )


        if result is None:
            print(
                f"[跳过] 无有效深度: {filename}"
            )
            continue


        abs_rel, mae, d1, norm_mae = result

        abs_rels.append(abs_rel)
        maes.append(mae)
        d1s.append(d1)
        norm_maes.append(norm_mae)


        # print(
        #     f"{filename}: "
        #     f"AbsRel={abs_rel:.4f}, "
        #     f"MAE={mae:.4f}, "
        #     f"d1={d1:.4f}"
        # )



    if len(abs_rels) == 0:
        raise RuntimeError(
            "没有有效样本参与评价"
        )


    return {
        "Mean Abs Rel": np.mean(abs_rels),
        "Mean MAE (m)": np.mean(maes),
        "Mean δ1": np.mean(d1s),
        "Mean Norm Error": np.nanmean(norm_maes)
    }



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument("--gt_dir", required=True, type=str)
    args = parser.parse_args()

    model_dir = args.model_dir
    gt_dir = args.gt_dir


    result = evaluate_all(
        model_dir,
        gt_dir
    )


    print("\n========== Average Depth Metrics ==========")

    for k, v in result.items():
        print(
            f"{k}: {v:.4f}"
        )