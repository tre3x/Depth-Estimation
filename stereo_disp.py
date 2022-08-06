from __future__ import print_function, division
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from stereo_depth_utils.datasets import dataset as Dataset
from stereo_depth_utils.models import __models__
from stereo_depth_utils.kitticolormap import *

class stereo_depth():
    def __init__(self, modeltype, imageLeft, imageRight, loadckpt, maxdisp=192, store = False):
        self.modeltype = modeltype
        self.maxdisp = maxdisp
        self.imageLeft = imageLeft
        self.imageRight = imageRight
        self.loadckpt = loadckpt
        self.store = store

    def tensor2numpy(self, vars):
        if isinstance(vars, np.ndarray):
            return vars
        elif isinstance(vars, torch.Tensor):
            return vars.data.cpu().numpy()
        else:
            raise NotImplementedError("invalid input type for tensor2numpy")

    def test(self):
        print("Generating the disparity maps...")
        here = os.path.dirname(os.path.abspath(__file__))
        if self.store:
            os.makedirs(os.path.join(here, "prediction_stereo"), exist_ok=True)


        disp_est_tn = self.test_sample(self.test_dataset)
        disp_est_np = self.tensor2numpy(disp_est_tn)
        left_filename = self.test_dataset["left_filename"]["left_filename"]


        fn = os.path.join(here, "prediction_stereo", "{}".format(left_filename))

        disp_est = kitti_colormap(disp_est_np[0])
        disp_est = cv2.bitwise_not(disp_est)
        disp_est = cv2.cvtColor(disp_est, cv2.COLOR_BGR2GRAY)
        disp_est = cv2.GaussianBlur(disp_est,(5,5), 0)
        if self.store:
            cv2.imwrite(fn, disp_est)
        print("Stereo Depth Estimated!")
        return disp_est

    def test_sample(self, sample):
        self.model.eval()
        left = sample['left']['left']
        right = sample['right']['right']
        left = left.reshape(-1, left.shape[0], left.shape[1], left.shape[2])
        right = right.reshape(-1, right.shape[0], right.shape[1], right.shape[2])
        disp_ests = self.model(left, right)
        return disp_ests[-1]

    def run(self):
        cudnn.benchmark = True

        self.test_dataset = Dataset.StereoDataset(self.imageLeft, self.imageRight, self.store)
        self.model = __models__[self.modeltype](self.maxdisp)
        self.model = nn.DataParallel(self.model)

        print("Loading model {}".format(self.loadckpt))
        state_dict = torch.load(self.loadckpt, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict['model'])

        disp = self.test()
        return disp            


if __name__ == '__main__':
    test(args)