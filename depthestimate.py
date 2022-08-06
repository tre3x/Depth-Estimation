import os
import cv2
import argparse
import subprocess
from rectification import rectify
from stereo_disp import stereo_depth as sd
from monocular_depth import DepthEstimate

def store_depth(depth, filename):
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "final_depth"), exist_ok=True)
    cv2.imwrite(os.path.join(here, "final_depth", filename), depth)

def HybridStereoDepth(leftimgpath, rightimgpath, stereo_model, monocular_encoder, stereo_ckpt_path, monocular_checkpoint_path):
    left_rect, right_rect = rectify(leftimgpath, rightimgpath)
    stereodepth = sd(stereo_model, left_rect, right_rect, stereo_ckpt_path).run()
    mono_left_depth = DepthEstimate(left_rect, monocular_encoder, monocular_checkpoint_path)
    mono_right_depth = DepthEstimate(right_rect, monocular_encoder, monocular_checkpoint_path)
    aggregrate = (stereodepth+mono_left_depth+mono_right_depth)*0.33
    return aggregrate

def run(leftimgpath, anchorimgpath, rightimgpath, stereo_model, monocular_encoder, stereo_ckpt_path, monocular_checkpoint_path):
    map1 = HybridStereoDepth(leftimgpath, anchorimgpath, stereo_model, monocular_encoder, stereo_ckpt_path, monocular_checkpoint_path)
    map2 = HybridStereoDepth(anchorimgpath, rightimgpath, stereo_model, monocular_encoder, stereo_ckpt_path, monocular_checkpoint_path)
    aggregrate = (map1+map2)*0.5
    store_depth(aggregrate, anchorimgpath.split('/')[-1])

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Depth Estimation from three image feeds.')

    parser.add_argument('--leftimgpath', required = True, help='What is the left image path?')
    parser.add_argument('--anchorimgpath', required = True, help='What is the anchor image path?')
    parser.add_argument('--rightimgpath', required = True, help='What is the right image path?')
    parser.add_argument('--stereo_model', default='MSNet2D', help='what is the stereo model name? (MSNet2D, MSNet3D)')
    parser.add_argument('--monocular_encoder', default='large07', help='what is the monocular depth model name? (large07, )')
    parser.add_argument('--stereo_ckpt_path', required = True, help='What is the path to saved stereo depth estimation model?')
    parser.add_argument('--monocular_checkpoint_path', required = True, help='What is the path to saved monocular depth estimation model?')

    args = parser.parse_args()
    
    run(args.leftimgpath, args.anchorimgpath, args.rightimgpath, args.stereo_model, args.monocular_encoder, args.stereo_ckpt_path, args.monocular_checkpoint_path)