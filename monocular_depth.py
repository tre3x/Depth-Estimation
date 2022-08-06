from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import os, sys, errno
import argparse
import time
import numpy as np
import cv2
from tqdm import tqdm

from monoculardepth_utils.dataloader_kittipred import NewDataLoader
from monoculardepth_utils.NewCRFDepth import NewCRFDepth

def flip_lr(image):
    assert image.dim() == 4, 'You need to provide a [B,C,H,W] image to flip'
    return torch.flip(image, [3])


def post_process_depth(depth, depth_flipped, method='mean'):
    B, C, H, W = depth.shape
    inv_depth_hat = flip_lr(depth_flipped)
    inv_depth_fused = fuse_inv_depth(depth, inv_depth_hat, method=method)
    xs = torch.linspace(0., 1., W, device=depth.device,
                        dtype=depth.dtype).repeat(B, C, H, 1)
    mask = 1.0 - torch.clamp(20. * (xs - 0.05), 0., 1.)
    mask_hat = flip_lr(mask)
    return mask_hat * depth + mask * inv_depth_hat + \
           (1.0 - mask - mask_hat) * inv_depth_fused

def fuse_inv_depth(inv_depth, inv_depth_hat, method='mean'):
    if method == 'mean':
        return 0.5 * (inv_depth + inv_depth_hat)
    elif method == 'max':
        return torch.max(inv_depth, inv_depth_hat)
    elif method == 'min':
        return torch.min(inv_depth, inv_depth_hat)
    else:
        raise ValueError('Unknown post-process method {}'.format(method))


def DepthEstimate(img, encoder, checkpoint_path, max_depth=192, store=False):
    here = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.dirname(checkpoint_path)
    sys.path.append(model_dir)
    
    dataloader = NewDataLoader(img)
    
    model = NewCRFDepth(version=encoder, inv_depth=False, max_depth=max_depth)
    model = torch.nn.DataParallel(model)
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()


    pred_depths = []
    with torch.no_grad():
        for _, sample in enumerate(dataloader.data):
            image = Variable(sample['image'])
            depth_est = model(image)
            post_process = True
            if post_process:
                image_flipped = flip_lr(image)
                depth_est_flipped = model(image_flipped)
                depth_est = post_process_depth(depth_est, depth_est_flipped)

            pred_depth = depth_est.cpu().numpy().squeeze()

            pred_depths.append(pred_depth)
    if store:
        if not os.path.exists(os.path.join(here, "monocular-depth")):
            os.makedirs(os.path.join(here, "monocular-depth"))
    if type(img) is np.ndarray:
        outpath = os.path.join(here, "monocular-depth", "test.png")
    else:
        outpath = os.path.join(here, "monocular-depth", img.split('/')[-1])
    if store:
        cv2.imwrite(outpath, pred_depths[0])
    return pred_depths[0]


if __name__ == '__main__':
    imgpath = '/home/tre3x/python/mitacs/Depth-Estimation/rectified_output/TOMMY2_L/TOMMY2_LL.png'
    encoder = 'large07'
    checkpoint_path = '/home/tre3x/python/mitacs/Depth-Estimation/monoculardepth_utils/networks/model_nyu.ckpt'
    max_depth = 192
    test(imgpath, encoder, checkpoint_path, max_depth)
