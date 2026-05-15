import random
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import numpy as np
import albumentations as A
import torchvision.transforms.functional as TF
import torch

class FaceAugmentor:
    def __init__(self):
        self.post_aug = A.Compose([
            A.ColorJitter(brightness=(0.3, 1.3), contrast=0.3, saturation=0.3, hue=0.3, p=1.0),
            A.PiecewiseAffine(scale=(0.02, 0.04), p=1.0),
            A.GaussNoise(p=1),
        ])

    def random_aspect_resize(self, img, flag=None, scale=None):
        img = torch.from_numpy(img).permute(2, 0, 1)
        # img: torch.Tensor [C,H,W]
        H, W = img.shape[-2:]
        if flag is None:
            flag = random.random()
        if scale is None:
            scale = random.uniform(1.0, 1.3)

        if flag < 0.5:
            scale_x = scale
            scale_y = 1.0
        else:
            scale_x = 1.0
            scale_y = scale
        
        new_W, new_H = int(W * scale_x), int(H * scale_y)
        img_resized = TF.resize(img, (new_H, new_W), antialias=True)
        # 中心裁剪/填充回原尺寸
        img_final = TF.center_crop(img_resized, (H, W))
        return img_final.permute(1,2,0).numpy()  # [H,W,C]

    def __call__(self, img, random_size=True, flag=None, scale=None):  # img: numpy RGB [H, W, 3]
        h, w = img.shape[:2]
        img_aug = img.copy()

        if random_size:
            img_aug = self.random_aspect_resize(img_aug, flag, scale)

        return self.post_aug(image=img_aug)["image"]