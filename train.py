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
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')


import torch
import torch.nn.functional as F
import torchvision
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim, compute_scale_and_shift, ScaleAndShiftInvariantLoss, l1_loss_mask, l2_loss
#from utils.depth_utils import estimate_depth
from gaussian_renderer import prefilter_voxel, render, network_gui
from utils.graphics_utils import surface_normal_from_depth, img_warping, depth_propagation, check_geometric_consistency, generate_edge_mask # 新增
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, load_pairs_relation, vis_depth, read_propagted_depth
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import imageio # 新增
import cv2 #新增
from prune import prune_list, calculate_v_imp_score, prune_list_calculate_v_imp_score
from utils.pose_utils import generate_pseudo_view, is_straight_motion
from utils.graphics_utils import getWorld2View2
from utils.general_utils import get_linear_noise_func
import random
import math


# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')


def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)

    #read the overlapping txt
    if opt.dataset == '360' and opt.depth_loss:
        pairs = load_pairs_relation(opt.pair_path)
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    viewpoint_stack = scene.getTrainCameras().copy()

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    

    # depth_loss_fn = ScaleAndShiftInvariantLoss(alpha=0.1, scales=1)
    propagated_iteration_begin = opt.propagated_iteration_begin 
    propagated_iteration_after = opt.propagated_iteration_after 
    after_propagated = False 
    propagation_dict = {} 
    for i in range(0, len(viewpoint_stack), 1): 
        propagation_dict[viewpoint_stack[i].image_name] = False


    gaussians.mask_prunning = np.full((gaussians.get_anchor.shape[0],), False)
    print(gaussians.get_anchor.size())

    for iteration in range(first_iter, opt.iterations + 1):
        
        # network gui not available in scaffold-gs yet
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        
        # Pick a random Camera
        # if not viewpoint_stack:
        #     viewpoint_stack = scene.getTrainCameras().copy()
        randidx = randint(0, len(viewpoint_stack)-1) 
        # if iteration > propagated_iteration_begin and iteration < propagated_iteration_after and after_propagated:
        #     randidx = propagated_view_index
        viewpoint_cam = viewpoint_stack[randidx] 

        if opt.depth_loss: 
            if opt.dataset == '360':
                src_idxs = pairs[randidx]
            else:
                # intervals = [-6, -3, 3, 6]
                if opt.dataset == 'waymo':
                    intervals = [-2, -1, 1, 2]
                elif opt.dataset == 'scannet':
                    intervals = [-10, -5, 5, 10]
                elif opt.dataset == 'free':
                    intervals = [-2, -1, 1, 2]
                src_idxs = [randidx+itv for itv in intervals if ((itv + randidx > 0) and (itv + randidx < len(viewpoint_stack)))]

        #propagate the gaussians first
        with torch.no_grad():
           if opt.depth_loss and iteration > propagated_iteration_begin and iteration < propagated_iteration_after and (iteration % opt.propagation_interval == 0 and not propagation_dict[viewpoint_cam.image_name]):
            # if opt.depth_loss and iteration > propagated_iteration_begin and iteration < propagated_iteration_after and (iteration % opt.propagation_interval == 0):

                old_count = gaussians.get_anchor.shape[0]
                
                propagation_dict[viewpoint_cam.image_name] = True

                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, 
                            return_normal=opt.normal_loss, return_opacity=False, return_depth=opt.depth_loss or opt.depth2normal_loss)

                projected_depth = render_pkg["render_depth"]

                # get the opacity that less than the threshold, propagate depth in these region
                if viewpoint_cam.sky_mask is not None:
                    sky_mask = viewpoint_cam.sky_mask.to(opacity_mask.device).to(torch.bool)
                else:
                    sky_mask = None
                torchvision.utils.save_image(viewpoint_cam.original_image, "cost/"+viewpoint_cam.image_name+"_"+str(iteration)+"gt.png")

                # get the propagated depth
                propagated_depth, normal = depth_propagation(viewpoint_cam, projected_depth, viewpoint_stack, src_idxs, opt.dataset, opt.patch_size)
                # cache the propagated_depth
                viewpoint_cam.depth = propagated_depth

                #transform normal to camera coordinate
                R_w2c = torch.tensor(viewpoint_cam.R.T).cuda().to(torch.float32)
                # R_w2c[:, 1:] *= -1
                normal = (R_w2c @ normal.view(-1, 3).permute(1, 0)).view(3, viewpoint_cam.image_height, viewpoint_cam.image_width)                
                valid_mask = propagated_depth != 300

                # calculate the abs rel depth error of the propagated depth and rendered depth & render color error
                render_depth = render_pkg['render_depth']
                abs_rel_error = torch.abs(propagated_depth - render_depth) / propagated_depth
                abs_rel_error_threshold = opt.depth_error_max_threshold - (opt.depth_error_max_threshold - opt.depth_error_min_threshold) * (iteration - propagated_iteration_begin) / (propagated_iteration_after - propagated_iteration_begin)
                # color error
                render_color = render_pkg['render']
                torchvision.utils.save_image(render_color, "cost/"+viewpoint_cam.image_name+"_"+str(iteration)+"color.png")

                color_error = torch.abs(render_color - viewpoint_cam.original_image)
                color_error = color_error.mean(dim=0).squeeze()
                error_mask = (abs_rel_error > abs_rel_error_threshold)

                # # calculate the photometric consistency
                ref_K = viewpoint_cam.K
                #c2w
                ref_pose = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()
                
                # calculate the geometric consistency
                geometric_counts = None
                for idx, src_idx in enumerate(src_idxs):
                    src_viewpoint = viewpoint_stack[src_idx]
                    #c2w
                    src_pose = src_viewpoint.world_view_transform.transpose(0, 1).inverse()
                    src_K = src_viewpoint.K

                    if src_viewpoint.depth is None:
                        src_render_pkg = render(src_viewpoint, gaussians, pipe, bg, 
                                return_normal=opt.normal_loss, return_opacity=False, return_depth=opt.depth_loss or opt.depth2normal_loss)
                        src_projected_depth = src_render_pkg['render_depth']
                    
                    #get the src_depth first
                        src_depth, src_normal = depth_propagation(src_viewpoint, src_projected_depth, viewpoint_stack, src_idxs, opt.dataset, opt.patch_size)
                        src_viewpoint.depth = src_depth
                    else:
                        src_depth = src_viewpoint.depth
                        
                    mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff = check_geometric_consistency(propagated_depth.unsqueeze(0), ref_K.unsqueeze(0), 
                                                                                                                 ref_pose.unsqueeze(0), src_depth.unsqueeze(0), 
                                                                                                                 src_K.unsqueeze(0), src_pose.unsqueeze(0), thre1=2, thre2=0.01)
                    
                    if geometric_counts is None:
                        geometric_counts = mask.to(torch.uint8)
                    else:
                        geometric_counts += mask.to(torch.uint8)
                        
                cost = geometric_counts.squeeze()
                cost_mask = cost >= 2       
                
                normal[~(cost_mask.unsqueeze(0).repeat(3, 1, 1))] = -10
                viewpoint_cam.normal = normal
                
                propagated_mask = valid_mask & error_mask & cost_mask
                if sky_mask is not None:
                    propagated_mask = propagated_mask & sky_mask

                propagated_depth[~cost_mask] = 300 
                # propagated_mask = propagated_mask & edge_mask
                propagated_depth[~propagated_mask] = 300

                if propagated_mask.sum() > 100:
                    down_sampling = None
                    if 0 < iteration <= 4000:
                        down_sampling = 16
                    elif 3000 < iteration <= 8000:
                        down_sampling = 8
                    elif 6000 < iteration <= 12000:
                        down_sampling = 4
                    gaussians.densify_from_depth_propagation(viewpoint_cam, propagated_depth, propagated_mask.to(torch.bool), gt_image, down_sampling)

                new_count = gaussians.get_anchor.shape[0]
                K = new_count - old_count
                new_mask = np.full((K,), True) 
                gaussians.mask_prunning = np.concatenate((gaussians.mask_prunning, new_mask), axis=0)

                """
                if gaussians.get_anchor.shape[0] == len(gaussians.mask_prunning):
                    print("get_anchor() 和 mask_prunning 的大小相等")
                else:
                    print("get_anchor() 和 mask_prunning 的大小不相等")
                """
        

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if args.augmented_view and iteration > 12000 and iteration % 2 == 0 and randidx > 0:
            # viewpoint_cam = gaussian_poses(viewpoint_cam, mean= 0, std_dev_translation=0.0001)
            prev_cam = viewpoint_stack[randidx - 1]
            if is_straight_motion(prev_cam, viewpoint_cam):
                viewpoint_cam = generate_pseudo_view(viewpoint_cam, std=0.01)

        
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe,background)

        # print(voxel_visible_mask)

        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, visible_mask=voxel_visible_mask, retain_grad=retain_grad, return_normal=opt.normal_loss, return_opacity=True, return_depth=opt.depth_loss or opt.depth2normal_loss)
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
        
        # opacity mask
        if iteration < opt.propagated_iteration_begin and opt.depth_loss:
            opacity_mask = render_pkg['render_opacity'] > 0.999
            opacity_mask = opacity_mask.unsqueeze(0).repeat(3, 1, 1)
        else:
            opacity_mask = render_pkg['render_opacity'] > 0.0
            opacity_mask = opacity_mask.unsqueeze(0).repeat(3, 1, 1)

        gt_image = viewpoint_cam.original_image.cuda()
        #Ll1 = l1_loss(image[opacity_mask], gt_image[opacity_mask])


        


        Ll1 = l1_loss(image, gt_image)

        #print(image.size())
        #print(gt_image.size())

        #ssim_loss = (1.0 - ssim(image, gt_image, mask=opacity_mask))
        ssim_loss = (1.0 - ssim(image, gt_image))

        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg

        # flatten loss
        if opt.flatten_loss: # 新增
            scales = scaling # 修改了，应该是这样
            min_scale, _ = torch.min(scales, dim=1)
            min_scale = torch.clamp(min_scale, 0, 30)
            flatten_loss = torch.abs(min_scale).mean()
            loss += opt.lambda_flatten * flatten_loss

        # opacity loss
        if opt.sparse_loss: # 新增
            opacity = opacity # 修改了，应该是这样
            opacity = opacity.clamp(1e-6, 1-1e-6)
            log_opacity = opacity * torch.log(opacity)
            log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
            sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[visibility_filter].mean()
            loss += opt.lambda_sparse * sparse_loss

        if opt.normal_loss:
            rendered_normal = render_pkg['render_normal']
            if viewpoint_cam.normal is not None:
                normal_gt = viewpoint_cam.normal.cuda()
                if viewpoint_cam.sky_mask is not None:
                    filter_mask = viewpoint_cam.sky_mask.to(normal_gt.device).to(torch.bool)
                    normal_gt[~(filter_mask.unsqueeze(0).repeat(3, 1, 1))] = -10
                filter_mask = (normal_gt != -10)[0, :, :].to(torch.bool)

                l1_normal = torch.abs(rendered_normal - normal_gt).sum(dim=0)[filter_mask].mean()
                cos_normal = (1. - torch.sum(rendered_normal * normal_gt, dim = 0))[filter_mask].mean()
                loss += opt.lambda_l1_normal * l1_normal + opt.lambda_cos_normal * cos_normal



        ###############AAAI depth loss
        if iteration > 12000 and iteration < 20000:
            # 提取天空掩码
            sky_mask_depth = torch.tensor(viewpoint_cam.sky_mask).cuda()
            #print(sky_mask_depth.shape)

            # 模型渲染输出的深度图，乘掩码去除天空
            rendered_depth = render_pkg["render_depth"] * sky_mask_depth
            #print(rendered_depth.shape)

            # 伪深度图（DepthAnything 输出），求通道平均，乘掩码去天空
            midas_depth = torch.tensor(viewpoint_cam.gt_depth).cuda()
            #print(midas_depth.shape)
            midas_depth = midas_depth.mean(dim=0) * sky_mask_depth  # (H, W)
            #print(midas_depth.shape)

            # 添加 batch 和 channel 维度
            if rendered_depth.dim() == 2:
                rendered_depth = rendered_depth.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            if midas_depth.dim() == 2:
                midas_depth = midas_depth.unsqueeze(0).unsqueeze(0)

            # 水平方向 Sobel 卷积核
            Sx = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(rendered_depth.device)

            # 计算梯度图
            Gdx = F.conv2d(rendered_depth, Sx, padding=1)
            Gdx_prime = F.conv2d(midas_depth, Sx, padding=1)

            # 像素级梯度差异图
            ldx_map = torch.abs(Gdx_prime - Gdx)  # [1, 1, H, W]

            # ===== 保留原始逻辑：拷贝梯度差图到 wdx_map =====
            wdx_map = ldx_map.clone()  # 保留原像素梯度差

            # ========= PATCH 权重图生成逻辑 ==========
            patch_size = 10
            H, W = ldx_map.shape[2], ldx_map.shape[3]
            ph, pw = H // patch_size, W // patch_size

            weight_map = torch.zeros_like(ldx_map)
            patch_means = []
            patch_indices = []

            for i in range(patch_size):
                for j in range(patch_size):
                    y0, y1 = i * ph, (i + 1) * ph
                    x0, x1 = j * pw, (j + 1) * pw

                    patch = midas_depth[:, :, y0:y1, x0:x1]
                    mean_depth = patch.mean()
                    patch_means.append(mean_depth)
                    patch_indices.append((y0, y1, x0, x1))

            patch_means_tensor = torch.stack(patch_means)  # shape: [patch_size * patch_size]
            sorted_indices = torch.argsort(patch_means_tensor, descending=True)  # 深度大的优先
            weights = 0.8 ** torch.arange(patch_size * patch_size).to(ldx_map.device)

            for rank, idx in enumerate(sorted_indices):
                mean_depth = patch_means_tensor[idx]
                y0, y1, x0, x1 = patch_indices[idx]
                if mean_depth == 0:
                    weight = 0.0
                else:
                    weight = weights[rank]
                weight_map[:, :, y0:y1, x0:x1] = weight

            # ===== 将 patch 权重乘到 wdx_map 上，形成最终加权图 =====
            wdx_map = wdx_map * weight_map
            # ========================================

            # 原始归一化与损失计算逻辑
            norm = torch.sqrt(torch.tensor(H * W, dtype=torch.float32)).to(rendered_depth.device)
            ldx = ldx_map.sum() / norm

            Ldg = (wdx_map * ldx).mean()
            loss += 0.05 * Ldg























        # if iteration > 12000 and iteration < 20000:
        #     # 提取天空掩码
        #     sky_mask_depth = torch.tensor(viewpoint_cam.sky_mask).cuda()

        #     # 模型渲染输出的深度图，乘掩码去除天空
        #     rendered_depth = render_pkg["render_depth"] * sky_mask_depth

        #     # 伪深度图（DepthAnything 输出），求通道平均，乘掩码去天空
        #     midas_depth = torch.tensor(viewpoint_cam.gt_depth).cuda()
        #     midas_depth = midas_depth.mean(dim=0) * sky_mask_depth  # (H, W)

        #     # 添加 batch 和 channel 维度
        #     if rendered_depth.dim() == 2:
        #         rendered_depth = rendered_depth.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        #     if midas_depth.dim() == 2:
        #         midas_depth = midas_depth.unsqueeze(0).unsqueeze(0)

        #     # 水平方向 Sobel 卷积核
        #     Sx = torch.tensor([[-1, 0, 1],
        #                     [-2, 0, 2],
        #                     [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(rendered_depth.device)

        #     # 计算梯度图
        #     Gdx = F.conv2d(rendered_depth, Sx, padding=1)
        #     Gdx_prime = F.conv2d(midas_depth, Sx, padding=1)

        #     # 像素级梯度差异图
        #     ldx_map = torch.abs(Gdx_prime - Gdx)  # [1, 1, H, W]

        #     # ===== 保留原始逻辑：拷贝梯度差图到 wdx_map =====
        #     wdx_map = ldx_map.clone()  # 保留原像素梯度差

        #     # ========= PATCH 权重图生成逻辑（按均值-区间-指数权重） ==========
        #     patch_size = 10
        #     H, W = ldx_map.shape[2], ldx_map.shape[3]
        #     ph, pw = H // patch_size, W // patch_size

        #     # 定义区间：0单独一档，(0,255]划10个区间
        #     bins = torch.linspace(0, 255, steps=11).to(midas_depth.device)  # [0,25.5,...,255]

        #     weight_map = torch.zeros_like(ldx_map)

        #     for i in range(patch_size):
        #         for j in range(patch_size):
        #             y0, y1 = i * ph, (i + 1) * ph
        #             x0, x1 = j * pw, (j + 1) * pw

        #             patch = midas_depth[:, :, y0:y1, x0:x1]  # [1,1,ph,pw]
        #             mean_depth = patch.mean()

        #             if mean_depth.item() == 0:
        #                 patch_weight = 0.0
        #             else:
        #                 idx = torch.bucketize(mean_depth, bins) - 1  # index: 0~9
        #                 idx = (9 - idx).clamp(0, 9).float()  # 深度大idx小，权重大
        #                 patch_weight = (0.3 ** idx)

        #             weight_map[:, :, y0:y1, x0:x1] = patch_weight

        #     # ===== 将 patch 权重乘到 wdx_map 上，形成最终加权图 =====
        #     wdx_map = wdx_map * weight_map
        #     # ========================================

        #     # 原始归一化与损失计算逻辑
        #     norm = torch.sqrt(torch.tensor(H * W, dtype=torch.float32)).to(rendered_depth.device)
        #     ldx = ldx_map.sum() / norm

        #     Ldg = (wdx_map * ldx).mean()
        #     loss += 0.05 * Ldg

        # if iteration>15000 and iteration<18000 or iteration>20000 and iteration<23000 or iteration>25000:
        #     sky_mask_depth = torch.tensor(viewpoint_cam.sky_mask).cuda()
        #     print(sky_mask_depth.shape)
        #     rendered_depth = torch.tensor(viewpoint_cam.rendered_depth).cuda()
        #     print(rendered_depth.shape)
        # if iteration>15000 and iteration<18000 or iteration>20000 and iteration<23000 or iteration>25000:
        #     sky_mask_depth = torch.tensor(viewpoint_cam.sky_mask).cuda()
        #     print(sky_mask_depth.shape)
        #     rendered_depth = render_pkg["render_depth"]*sky_mask_depth
        #     midas_depth = torch.tensor(viewpoint_cam.gt_depth).cuda()
        #     print(midas_depth.shape)
        #     midas_depth = midas_depth.mean(dim=0)*sky_mask_depth#
        #     # 添加 batch 和 channel 维度
        #     if rendered_depth.dim() == 2:
        #         rendered_depth = rendered_depth.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        #     if midas_depth.dim() == 2:
        #         midas_depth = midas_depth.unsqueeze(0).unsqueeze(0)

        #     # 定义 Sx
        #     Sx = torch.tensor([[-1, 0, 1],
        #                     [-2, 0, 2],
        #                     [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(rendered_depth.device)

        #     # 卷积
        #     Gdx = F.conv2d(rendered_depth, Sx, padding=1)
        #     Gdx_prime = F.conv2d(midas_depth, Sx, padding=1)

        #     # 差异图
        #     ldx_map = torch.abs(Gdx_prime - Gdx)
        #     wdx_map = ldx_map.clone()

        #     # 归一化
        #     H, W = rendered_depth.shape[2], rendered_depth.shape[3]
        #     norm = torch.sqrt(torch.tensor(H * W, dtype=torch.float32)).to(rendered_depth.device)
        #     ldx = ldx_map.sum() / norm

        #     # 最终加权损失
        #     Ldg = (wdx_map * ldx).mean()
        #     loss += 0.00005 * Ldg
        
        
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if not torch.isnan(loss):
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                
                # densification
                # 再看看这里需不需要改
                #if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    #gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
                

                ###
                if iteration > opt.update_from and iteration <= 15000 and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor_wait(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity, mask_prunning = gaussians.mask_prunning)
                if iteration >= 17000 and iteration <= 20000 and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor_wait(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity, mask_prunning = gaussians.mask_prunning)
                if iteration >= 22000 and iteration <= 25000 and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor_wait(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity, mask_prunning = gaussians.mask_prunning)
                

                ###
                #if iteration > opt.update_from and iteration <= 25000 and iteration % opt.update_interval == 0:
                    #gaussians.adjust_anchor_wait(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity, mask_prunning = gaussians.mask_prunning)
            
            elif iteration == opt.update_until:
                #del gaussians.opacity_accum
                #del gaussians.offset_gradient_accum
                #del gaussians.offset_denom
                torch.cuda.empty_cache()



            if iteration in args.prune_iterations:
                print(gaussians.get_anchor.size())
                # TODO Add prunning types
                gaussian_list, v_list = prune_list_calculate_v_imp_score(gaussians, scene, pipe, background)
                print(v_list)
                i = args.prune_iterations.index(iteration)
                gaussians.prune_gaussians(
                    (args.prune_decay**i) * args.prune_percent, v_list, gaussians.mask_prunning
                )
                print(gaussians.get_anchor.size())

            # if iteration % 1000 == 0 and iteration >= 2000 and iteration <= 11000:
            #     print(gaussians.get_anchor.shape[0])
            #     gaussians.voxelize_true_mask(voxel_size=0.001)
            #     print(gaussians.get_anchor.shape[0])
            	# voxelize propagated anchors every 1000 iters (paper eq. 6, 8)
            tau = 12000
            eps_b = 0.01
            eps_f = 0.005
            if iteration % 1000 == 0 and 2000 <= iteration <= tau:
                progress = iteration / tau
                voxel_size = max(
                    math.floor((eps_b - (eps_b - eps_f) * progress) * 1000) / 1000,
                    eps_f,
                )
                if gaussians.mask_prunning.any():
                    print(gaussians.get_anchor.shape[0])
                    gaussians.voxelize_true_mask(voxel_size=voxel_size)
                    print(gaussians.get_anchor.shape[0])


            """
            if gaussians.get_anchor.shape[0] == len(gaussians.mask_prunning):
                print("get_anchor() 和 mask_prunning 的大小真相等")
            else:
                print("get_anchor() 和 mask_prunning 的大小真不相等")
            """



            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                # 新增
                
                #if iteration == checkpoint_iterations[-1]:
                    #v_list = prune_list_calculate_v_imp_score(gaussians, scene, pipe, background, args.v_pow)
                    #np.savez(os.path.join(scene.model_path,"imp_score"), v_list.cpu().detach().numpy())


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)


    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if tb_writer:
            # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

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
        # 将天空区域设置为300
        if view.sky_mask is not None:
            render_depth[~(view.sky_mask.to(render_depth.device).to(torch.bool))] = 300
        # 保存原始 float32 深度数组（例如 .npy 格式）
        raw_depth_np = render_depth.detach().cpu().numpy()
        np.save(os.path.join(depth_path, '{0:05d}.npy'.format(idx)), raw_depth_np)
        # 再可视化成伪彩色图像
        render_depth_vis = vis_depth(raw_depth_np)[0]
        imageio.imwrite(os.path.join(depth_path , '{0:05d}.png'.format(idx)), render_depth_vis)

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

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "fps.txt"), 'w') as f:
        f.write(f"{fps:.5f}\n")

    
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


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '1')
    parser.add_argument("--augmented_view", action="store_true")
    parser.add_argument("--sample", action="store_true")
    
    parser.add_argument(
        "--prune_iterations", nargs="+", type=int, default=[15_000, 20_000]
    )
    parser.add_argument("--prune_percent", type=float, default=0.1)
    parser.add_argument("--v_pow", type=float, default=0.1)
    parser.add_argument("--prune_decay", type=float, default=0.8)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    
    # enable logging
    
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    

    #try:
        #saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    #except:
        #logger.info(f'save code failed~')
        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training
    training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger)



    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path)

    # All done
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f'\nStarting Rendering~')
    visible_count = render_sets(lp.extract(args), -1, pp.extract(args), wandb=wandb, logger=logger)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")
