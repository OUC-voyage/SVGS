"""
Align Depth Anything V2 grayscale depth to the COLMAP scene scale.

The grayscale PNG is the official --grayscale output: one per-image min-max
of the network disparity, not an inverted depth. Near surfaces are bright.
Each image is fit independently:

    z = 1 / (a * d + b),  d = gray / 255

COLMAP points only estimate a and b. The saved map is the dense aligned depth.
The input depth folder is only read.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_next_bytes,
)

DEPTH_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def resolve_model_dir(colmap_dir):
    candidates = [
        colmap_dir,
        os.path.join(colmap_dir, "0"),
        os.path.join(colmap_dir, "sparse", "0"),
        os.path.join(colmap_dir, "sparse"),
    ]
    needed = ("cameras", "images", "points3D")
    for folder in candidates:
        if not os.path.isdir(folder):
            continue
        if all(
            os.path.isfile(os.path.join(folder, name + ".bin"))
            or os.path.isfile(os.path.join(folder, name + ".txt"))
            for name in needed
        ):
            return folder
    raise FileNotFoundError(
        "COLMAP folder must contain cameras, images and points3D "
        "(.bin or .txt). Looked in: " + ", ".join(candidates)
    )


def read_model(model_dir):
    def read_pair(stem, read_bin, read_txt):
        bin_path = os.path.join(model_dir, stem + ".bin")
        txt_path = os.path.join(model_dir, stem + ".txt")
        if os.path.isfile(bin_path):
            return read_bin(bin_path)
        if os.path.isfile(txt_path):
            return read_txt(txt_path)
        raise FileNotFoundError(stem)

    images = read_pair("images", read_extrinsics_binary, read_extrinsics_text)
    cameras = read_pair("cameras", read_intrinsics_binary, read_intrinsics_text)
    points = read_points3d(model_dir)
    return images, cameras, points


def read_points3d(model_dir):
    bin_path = os.path.join(model_dir, "points3D.bin")
    txt_path = os.path.join(model_dir, "points3D.txt")
    id_to_xyz = {}
    if os.path.isfile(bin_path):
        with open(bin_path, "rb") as fid:
            num_points = read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_points):
                props = read_next_bytes(fid, 43, "QdddBBBd")
                track_length = read_next_bytes(fid, 8, "Q")[0]
                if track_length:
                    read_next_bytes(fid, 8 * int(track_length), "ii" * int(track_length))
                id_to_xyz[int(props[0])] = np.array(props[1:4], dtype=np.float64)
        return id_to_xyz
    if os.path.isfile(txt_path):
        with open(txt_path, "r") as fid:
            for line in fid:
                line = line.strip()
                if not line or line[0] == "#":
                    continue
                elems = line.split()
                id_to_xyz[int(elems[0])] = np.array(elems[1:4], dtype=np.float64)
        return id_to_xyz
    raise FileNotFoundError("points3D.bin / points3D.txt")


def index_images_by_stem(images):
    by_stem = {}
    for image in images.values():
        stem = os.path.splitext(os.path.basename(image.name))[0]
        by_stem.setdefault(stem, []).append(image)
    return by_stem


def load_disparity(path):
    array = np.array(Image.open(path))
    if array.ndim == 3:
        array = array[..., 0]
    if array.dtype == np.uint8:
        disparity = array.astype(np.float64) / 255.0
    elif array.dtype == np.uint16:
        disparity = array.astype(np.float64) / 65535.0
    else:
        disparity = array.astype(np.float64)
        peak = float(np.nanmax(disparity)) if disparity.size else 1.0
        if peak > 1.5:
            disparity = disparity / peak
    return disparity


def collect_correspondences(image, camera, points, disparity):
    height, width = disparity.shape
    scale_x = width / float(camera.width)
    scale_y = height / float(camera.height)
    rotation_c2w = np.transpose(qvec2rotmat(image.qvec))
    translation = np.asarray(image.tvec, dtype=np.float64)

    disparities = []
    depths = []
    point_ids = image.point3D_ids
    pixels = image.xys
    if point_ids is None or len(point_ids) == 0:
        return np.empty(0), np.empty(0)

    for pixel, point_id in zip(pixels, point_ids):
        point_id = int(point_id)
        if point_id < 0 or point_id not in points:
            continue
        camera_xyz = rotation_c2w.T @ points[point_id] + translation
        z = float(camera_xyz[2])
        if z <= 0.1:
            continue
        u = int(round(float(pixel[0]) * scale_x))
        v = int(round(float(pixel[1]) * scale_y))
        if u < 0 or v < 0 or u >= width or v >= height:
            continue
        d = float(disparity[v, u])
        if not np.isfinite(d) or d <= 1.0 / 255.0:
            continue
        disparities.append(d)
        depths.append(z)
    return np.asarray(disparities), np.asarray(depths)


def fit_inverse(disparities, depths, rng, iterations, rel_thresh):
    count = disparities.shape[0]
    if count < 2:
        return None
    best_inliers = None
    best_count = 0
    for _ in range(iterations):
        sample = rng.choice(count, 2, replace=False)
        d0, d1 = disparities[sample]
        if abs(d0 - d1) < 1e-6:
            continue
        design = np.stack([disparities[sample], np.ones(2)], axis=1)
        try:
            coeff, _, _, _ = np.linalg.lstsq(design, 1.0 / depths[sample], rcond=None)
        except np.linalg.LinAlgError:
            continue
        a, b = float(coeff[0]), float(coeff[1])
        denom = a * disparities + b
        predicted = np.full(count, np.inf)
        positive = denom > 1e-8
        predicted[positive] = 1.0 / denom[positive]
        rel = np.abs(predicted - depths) / depths
        inliers = positive & np.isfinite(rel) & (rel < rel_thresh)
        inlier_count = int(inliers.sum())
        if inlier_count > best_count:
            best_count = inlier_count
            best_inliers = inliers
    if best_inliers is None or best_count < 2:
        return None
    design = np.stack([disparities[best_inliers], np.ones(best_count)], axis=1)
    coeff, _, _, _ = np.linalg.lstsq(design, 1.0 / depths[best_inliers], rcond=None)
    a, b = float(coeff[0]), float(coeff[1])
    denom = a * disparities[best_inliers] + b
    predicted = 1.0 / denom
    rel = np.abs(predicted - depths[best_inliers]) / depths[best_inliers]
    return a, b, best_inliers, float(np.median(rel))


def apply_scale(disparity, a, b):
    aligned = np.zeros(disparity.shape, dtype=np.float32)
    denom = a * disparity + b
    valid = np.isfinite(denom) & (denom > 1e-8)
    values = np.zeros(disparity.shape, dtype=np.float64)
    values[valid] = 1.0 / denom[valid]
    valid &= np.isfinite(values) & (values > 0)
    aligned[valid] = values[valid].astype(np.float32)
    return aligned


def list_depth_files(depth_dir):
    files = []
    for name in sorted(os.listdir(depth_dir)):
        path = os.path.join(depth_dir, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in DEPTH_EXTS:
            files.append(path)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Align Depth Anything V2 grayscale depth to COLMAP scale."
    )
    parser.add_argument("--depth_dir", required=True, help="Input depth folder. It is not modified.")
    parser.add_argument("--colmap_dir", required=True, help="COLMAP folder, usually sparse/0.")
    parser.add_argument("--output_dir", required=True, help="Output folder, e.g. the sibling depth_scale.")
    parser.add_argument("--min_points", type=int, default=30, help="Minimum inliers required to keep an image.")
    parser.add_argument("--rel_thresh", type=float, default=0.15, help="RANSAC relative depth error.")
    parser.add_argument("--ransac_iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    depth_dir = os.path.abspath(args.depth_dir)
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(depth_dir):
        raise FileNotFoundError(depth_dir)
    if os.path.normcase(depth_dir) == os.path.normcase(output_dir):
        raise ValueError("output_dir must be different from depth_dir.")
    if os.path.normcase(output_dir).startswith(os.path.normcase(depth_dir + os.sep)):
        raise ValueError("output_dir must not be inside depth_dir.")

    model_dir = resolve_model_dir(os.path.abspath(args.colmap_dir))
    images, cameras, points = read_model(model_dir)
    by_stem = index_images_by_stem(images)
    depth_files = list_depth_files(depth_dir)
    if not depth_files:
        raise FileNotFoundError("No depth images in " + depth_dir)

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    records = []
    written = 0
    skipped = 0

    print("COLMAP: {}".format(model_dir))
    print("points: {}   images: {}   depth files: {}".format(len(points), len(images), len(depth_files)))

    for depth_path in depth_files:
        stem = os.path.splitext(os.path.basename(depth_path))[0]
        matches = by_stem.get(stem, [])
        record = {"name": os.path.basename(depth_path), "stem": stem}
        if len(matches) != 1:
            record["status"] = "no_colmap_image" if not matches else "ambiguous_name"
            records.append(record)
            skipped += 1
            print("skip {} ({})".format(stem, record["status"]))
            continue

        image = matches[0]
        camera = cameras[image.camera_id]
        disparity = load_disparity(depth_path)
        observed_d, observed_z = collect_correspondences(image, camera, points, disparity)
        record["correspondences"] = int(observed_d.shape[0])
        if observed_d.shape[0] < args.min_points:
            record["status"] = "too_few_points"
            records.append(record)
            skipped += 1
            print("skip {} (points {})".format(stem, observed_d.shape[0]))
            continue

        fitted = fit_inverse(observed_d, observed_z, rng, args.ransac_iter, args.rel_thresh)
        if fitted is None:
            record["status"] = "fit_failed"
            records.append(record)
            skipped += 1
            print("skip {} (fit failed)".format(stem))
            continue

        a, b, inliers, median_rel = fitted
        if int(inliers.sum()) < args.min_points:
            record["status"] = "too_few_inliers"
            record.update({"a": a, "b": b, "inliers": int(inliers.sum()), "median_rel": median_rel})
            records.append(record)
            skipped += 1
            print("skip {} (inliers {})".format(stem, int(inliers.sum())))
            continue

        aligned = apply_scale(disparity, a, b)
        out_path = os.path.join(output_dir, stem + ".npy")
        np.save(out_path, aligned)
        valid = aligned[aligned > 0]
        record.update({
            "status": "ok",
            "a": a,
            "b": b,
            "inliers": int(inliers.sum()),
            "median_rel": median_rel,
            "valid_pixels": int(valid.size),
            "z_min": float(valid.min()) if valid.size else 0.0,
            "z_median": float(np.median(valid)) if valid.size else 0.0,
            "z_max": float(valid.max()) if valid.size else 0.0,
            "output": out_path,
        })
        records.append(record)
        written += 1
        print(
            "ok   {}  points {}  inliers {}  a {:.6g}  b {:.6g}  median_rel {:.4f}  z [{:.3f}, {:.3f}]".format(
                stem,
                observed_d.shape[0],
                int(inliers.sum()),
                a,
                b,
                median_rel,
                record["z_min"],
                record["z_max"],
            )
        )

    log_path = os.path.join(output_dir, "scales.json")
    with open(log_path, "w", encoding="utf-8") as fid:
        json.dump(
            {
                "depth_dir": depth_dir,
                "colmap_dir": model_dir,
                "output_dir": output_dir,
                "written": written,
                "skipped": skipped,
                "images": records,
            },
            fid,
            indent=2,
        )
    print("written {}   skipped {}   log {}".format(written, skipped, log_path))


if __name__ == "__main__":
    main()
