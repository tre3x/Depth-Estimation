import os
import cv2
import argparse
import subprocess
from rectification import rectify
from stereo_disp import stereo_depth as sd

if __name__=='__main__':
    IMG_PATHL = '/home/at02400@ens.ad.etsmtl.ca/depth-estimation/rectification/uncalibrated_stereo_rectification/data/TOMMY2_L.png'
    IMG_PATHR = '/home/at02400@ens.ad.etsmtl.ca/depth-estimation/rectification/uncalibrated_stereo_rectification/data/TOMMY2_R.png'
    stereo_dataset = 'drivingstereo'
    stereo_model = 'MSNet2D'
    stereo_ckpt_path = '/home/at02400@ens.ad.etsmtl.ca/depth-estimation/pipeline/stereo_depth_utils/checkpoint/MSNet2D_SF_DS_KITTI2015.ckpt'

    here = os.path.dirname(os.path.abspath(__file__))

    rectify(IMG_PATHL, IMG_PATHR)
    for subdir, dirs, files in os.walk(os.path.join(here, "rectified_output")):
        for dirr in dirs:
            leftimg = os.path.join(subdir, "{}".format(dirr), "{}L.png".format(dirr))
            rightimg = os.path.join(subdir, "{}".format(dirr), "{}R.png".format(dirr))
            sd(stereo_model, stereo_dataset, leftimg, rightimg, stereo_ckpt_path, dir).run()
