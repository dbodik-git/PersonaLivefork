import json
import os
import random
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
from pathlib import Path
from .utils import (
    crop_square_containing_face_patch, 
    get_bbox_param,
    get_mask,
)
from .face_augmentor import FaceAugmentor

class PortraitImageDataset(Dataset):
    def __init__(
        self,
        img_size,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        data_meta_paths=["./data/fahsion_meta.json"],
        sample_margin=30,
    ):
        super().__init__()

        self.img_size = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.sample_margin = sample_margin

        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    self.img_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.expression_transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.Resize(
                    (256, 256),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    self.img_size,
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ]
        )


        self.face_augmentor = FaceAugmentor()

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)
    
    def __getitem__(self, index):
        try:
            video_meta = self.vid_meta[index]
            video_path = video_meta["video_path"]

            
            kps_path = video_path.replace("videos", "boxes").replace(".mp4", ".pt")
            video_reader = VideoReader(video_path)
            landmarks = torch.load(kps_path)

            video_length = min(len(video_reader), len(landmarks))
        
            margin = self.sample_margin
            ref_img_idx = random.randint(0, video_length - 1)
            if video_length > 2 * margin + 4:
                left = max(0, ref_img_idx - margin)
                right = min(video_length, ref_img_idx + margin)

                # 拼接两个不重叠的区间
                part = list(range(0, left)) + list(range(right, video_length))
                tgt_img_idx = random.choice(part)
            else:
                tgt_img_idx = random.randint(0, video_length - 1)

            ref_img = video_reader[ref_img_idx]
            tgt_img = video_reader[tgt_img_idx]
            ref_face_box = landmarks[ref_img_idx]['face']
            tgt_face_box = landmarks[tgt_img_idx]['face']
            tgt_left_eye_box = landmarks[tgt_img_idx]['left_eye']
            tgt_right_eye_box = landmarks[tgt_img_idx]['right_eye']
            tgt_mouth_box = landmarks[tgt_img_idx]['mouth']
                
            tgt_face_patch = crop_square_containing_face_patch(tgt_img.asnumpy(), tgt_face_box)
            bbox_params = torch.from_numpy(get_bbox_param(tgt_face_box, ref_face_box))

            face_mask, local_mask = get_mask(tgt_face_box, tgt_left_eye_box, tgt_right_eye_box, tgt_mouth_box, side=512)

            tgt_face_patch = self.face_augmentor(tgt_face_patch)

            ref_img_pil = Image.fromarray(ref_img.asnumpy())
            tgt_img_pil = Image.fromarray(tgt_img.asnumpy())
            tgt_face_pil = Image.fromarray(tgt_face_patch)
            face_mask = Image.fromarray(face_mask)
            local_mask = Image.fromarray(local_mask)

            state = torch.get_rng_state()
            tgt_img = self.augmentation(tgt_img_pil, self.transform, state=state)
            ref_img = self.augmentation(ref_img_pil, self.transform, state=state)
            tgt_face = self.augmentation(tgt_face_pil, self.expression_transform, state=state)
            tgt_pose = self.augmentation(tgt_img_pil, self.cond_transform, state=state)
            face_mask = self.augmentation(face_mask, self.mask_transform, state=state)
            local_mask = self.augmentation(local_mask, self.mask_transform, state=state)

            clip_image = self.clip_image_processor(
                images=ref_img_pil, return_tensors="pt"
            ).pixel_values[0]

            sample = dict(
                video_dir=video_path,
                img=tgt_img,
                ref_img=ref_img,
                tgt_face=tgt_face,
                tgt_pose=tgt_pose,
                bbox_params=bbox_params,
                clip_image=clip_image,
                face_mask=face_mask,
                local_mask=local_mask,
            )
            return sample
        except Exception as e:
            next_index = random.randint(0, len(self.vid_meta)-1)
            return self.__getitem__(next_index)
        
    def __len__(self):
        return len(self.vid_meta)