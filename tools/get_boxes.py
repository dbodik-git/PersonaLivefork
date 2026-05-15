import mediapipe as mp
import cv2
import os
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from src.utils.util import read_frames
import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)
import argparse
from functools import partial

mp_face_mesh = mp.solutions.face_mesh

face_indices = list(range(468))
left_eye_indices = [226, 230, 223, 245]
right_eye_indices = [446, 450, 465, 443]
mouth_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291] + \
                [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308] + \
                [146, 91, 181, 84, 17, 314, 405, 321, 375] + \
                [191, 80, 81, 82, 13, 312, 311, 310, 415]

def get_region_box(landmarks, indices):
    xs = [int(landmarks[i][0]) for i in indices]
    ys = [int(landmarks[i][1]) for i in indices]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max, y_max]

def process_video(name, video_dir, save_dir):
    video_path = os.path.join(video_dir, name)
    save_path = os.path.join(save_dir, name.replace('.mp4', '.pt'))
    if os.path.exists(save_path):
        return name
        
    video = read_frames(video_path)
    boxes = []

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        for image_pil in video:
            image = np.array(image_pil)
            h, w, _ = image.shape
            results = face_mesh.process(image)

            if results.multi_face_landmarks is not None:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = [(int(l.x * w), int(l.y * h)) for l in face_landmarks.landmark]
                face_box = get_region_box(landmarks, face_indices)
                left_eye_box = get_region_box(landmarks, left_eye_indices)
                right_eye_box = get_region_box(landmarks, right_eye_indices)
                mouth_box = get_region_box(landmarks, mouth_indices)
                boxes.append({
                    'face': face_box,
                    'left_eye': left_eye_box,
                    'right_eye': right_eye_box,
                    'mouth': mouth_box
                })
            else:
                boxes.append(boxes[-1] if boxes else {
                    'face': [],
                    'left_eye': [],
                    'right_eye': [],
                    'mouth': []
                })
    torch.save(boxes, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract facial boxes from videos and save as .pt files.")
    parser.add_argument('--video_dir', type=str, default='/home/zyli/Repositories/x-nemo-inference/lv100/videos')
    parser.add_argument('--save_dir', type=str, default='/home/zyli/Repositories/x-nemo-inference/lv100/boxes_zyli')
    parser.add_argument('--workers', type=int, default=8)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    video_files = [f for f in os.listdir(args.video_dir) if f.endswith('.mp4')]

    process_func = partial(process_video, video_dir=args.video_dir, save_dir=args.save_dir)

    with ProcessPoolExecutor(max_workers=8) as executor:
        list(executor.map(process_func, video_files))