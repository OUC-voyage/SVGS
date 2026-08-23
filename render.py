#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import torch

import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_depth") # 新增
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_normal") # 新增

    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path , exist_ok=True) # 新增
    makedirs(normal_path, exist_ok=True) # 新增
    
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, return_depth=True, return_normal=True, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)


        # gts
        gt = view.original_image[0:3, :, :]
        
        # error maps
        errormap = (rendering - gt).abs()


        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item() # 将每个视图的可见物体数量记录在 per_view_dict 中，键为文件名，值为可见物体数量。

        # 新增从这到for循环结束,本来这段在with open的上面。
        render_depth = render_pkg["render_depth"]
        if view.sky_mask is not None:
            render_depth[~(view.sky_mask.to(render_depth.device).to(torch.bool))] = 300
        render_depth = vis_depth(render_depth.detach().cpu().numpy())[0]
        imageio.imwrite(os.path.join(depth_path , '{0:05d}'.format(idx) + ".png"), render_depth)



        # ===== 新增：DepthAnything 风格的近亮远暗灰度深度图 =====

        # 注意：这里重新从 render_pkg 取一次，避免被 vis_depth 覆盖
        render_depth_nb = render_pkg["render_depth"].clone()

        if view.sky_mask is not None:
            render_depth_nb[~(view.sky_mask.to(render_depth_nb.device).to(torch.bool))] = 300.0

        d = render_depth_nb.detach().cpu().numpy()

        # min-max 归一化
        d_min = d.min()
        d_max = d.max()
        d_norm = (d - d_min) / (d_max - d_min + 1e-8)

        # 反转：近亮远暗
        d_inv = 1.0 - d_norm

        # 映射到 0–255
        depth_gray = (d_inv * 255.0).clip(0, 255).astype("uint8")
        print(depth_gray.min(), depth_gray.max())

        # 用不同文件名，避免覆盖
        imageio.imwrite(
            os.path.join(depth_path, f"{idx:05d}_nearbright.png"),
            depth_gray
        )



        render_normal = (render_pkg["render_normal"] + 1.0) / 2.0
        if view.sky_mask is not None:
            render_normal[~(view.sky_mask.to(rendering.device).to(torch.bool).unsqueeze(0).repeat(3, 1, 1))] = -10
        # render_normal = renders["render_normal"]
        np.save(os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"), render_pkg["render_normal"].detach().cpu().numpy())
        torchvision.utils.save_image(render_normal, os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"))
        # normal_gt = torch.nn.functional.normalize(view.normal, p=2, dim=0)
        # render_normal_gt = (normal_gt + 1.0) / 2.0
        # torchvision.utils.save_image(render_normal_gt, os.path.join(normal_path, '{0:05d}'.format(idx) + "_normalgt.png"))
        # exit()
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list

    

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        # scales = gaussians.get_scaling # 新增

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    return visible_count

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
