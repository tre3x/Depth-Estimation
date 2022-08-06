import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

class StereoDataset(Dataset):
    def __init__(self, left_image, right_image, store):
        self.left_image = left_image
        self.right_image = right_image
        self.left_filename = 'test.png'
        if store and type(self.left_image) is not np.ndarray:
            self.left_filename = self.left_image.split('/')[-1]

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def __getitem__(self, index):
        if type(self.left_image) is not np.ndarray and type(self.right_image) is not np.ndarray:
            left_img = self.load_image(self.left_image)
            right_img = self.load_image(self.right_image)
        else:
            left_img = Image.fromarray(self.left_image.astype('uint8'), 'RGB')
            right_img = Image.fromarray(self.right_image.astype('uint8'), 'RGB')

        w, h = left_img.size
        crop_w, crop_h = 880, 400

        left_img = left_img.resize((1248, 384), Image.ANTIALIAS)
        right_img = right_img.resize((1248, 384), Image.ANTIALIAS)

        processed = get_transform()
        left_img = processed(left_img)
        right_img = processed(right_img)

        return {"left": left_img,
                "right": right_img,
                "top_pad": 0,
                "right_pad": 0,
                "left_filename": self.left_filename}
