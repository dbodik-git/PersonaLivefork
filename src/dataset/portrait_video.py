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
import gc
from .face_augmentor import FaceAugmentor
import torchvision.transforms.functional as TF

class PortraitVideoDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        n_sample_frames,
        img_size,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        data_meta_paths=["./data/fahsion_meta.json"],
    ):
        super().__init__()

        self.img_size = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames

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

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    self.img_size,
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ]
        )

        self.pose_transform = transforms.Compose(
            [
                transforms.Resize(
                    (256, 256),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

        self.face_augmentor = FaceAugmentor()

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor
    
    def __getitem__(self, index):
        try:
            video_meta = self.vid_meta[index]
            video_path = video_meta["video_path"]

            
            kps_path = video_path.replace("videos", "boxes").replace(".mp4", ".pt")
            video_reader = VideoReader(video_path)
            landmarks = torch.load(kps_path)

            video_length = min(len(video_reader), len(landmarks))
        
            ref_img_idx = random.randint(0, video_length - 1)
            sample_rate = self.sample_rate
            clip_length = min(video_length, (self.n_sample_frames - 1) * sample_rate + 1)
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int).tolist()

            ref_img = video_reader[ref_img_idx]
            ref_face_box = landmarks[ref_img_idx]['face']

            tgt_image_list = []
            tgt_face_list = []
            tgt_face_mask_list = []
            tgt_local_mask_list = []
            bbox_params = []

            face_flag = random.random()
            face_scale = random.uniform(1.0, 1.2)

            for tgt_img_idx in batch_index:
                tgt_img = video_reader[tgt_img_idx]
                tgt_face_box = landmarks[tgt_img_idx]['face']
                tgt_left_eye_box = landmarks[tgt_img_idx]['left_eye']
                tgt_right_eye_box = landmarks[tgt_img_idx]['right_eye']
                tgt_mouth_box = landmarks[tgt_img_idx]['mouth']
                
                tgt_face_patch = crop_square_containing_face_patch(tgt_img.asnumpy(), tgt_face_box)
                face_mask, local_mask = get_mask(tgt_face_box, tgt_left_eye_box, tgt_right_eye_box, tgt_mouth_box, side=512)
                tgt_face_patch = self.face_augmentor(tgt_face_patch, flag=face_flag, scale=face_scale)

                bbox_param = torch.from_numpy(get_bbox_param(tgt_face_box, ref_face_box))
                bbox_params.append(bbox_param)

                tgt_image_list.append(Image.fromarray(tgt_img.asnumpy()))
                tgt_face_list.append(Image.fromarray(tgt_face_patch))
                tgt_face_mask_list.append(Image.fromarray(face_mask))
                tgt_local_mask_list.append(Image.fromarray(local_mask))

            bbox_params = torch.stack(bbox_params, dim=0)
            ref_img_pil = Image.fromarray(ref_img.asnumpy())
            
            state = torch.get_rng_state()
            tgt_img = self.augmentation(tgt_image_list, self.transform, state=state)
            ref_img = self.augmentation(ref_img_pil, self.transform, state=state)
            tgt_face = self.augmentation(tgt_face_list, self.expression_transform, state=state)
            face_mask = self.augmentation(tgt_face_mask_list, self.mask_transform, state=state)
            local_mask = self.augmentation(tgt_local_mask_list, self.mask_transform, state=state)

            tgt_pose = self.augmentation(tgt_image_list, self.pose_transform, state=state)

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