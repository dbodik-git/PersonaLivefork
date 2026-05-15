import argparse
import json
import os

# -----
# python tools/extract_meta_info.py --root_path /path/to/video_dir --dataset_name fashion
# -----
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default='./Datasets/videos/VFHQ')
parser.add_argument("--dataset_name", type=str, default='VFHQ')
parser.add_argument("--meta_info_name", type=str)

args = parser.parse_args()

if args.meta_info_name is None:
    args.meta_info_name = args.dataset_name

# collect all video_folder paths
video_mp4_paths = set()
for root, dirs, files in os.walk(args.root_path):
    for name in files:
        if name.endswith(".pt"):
            video_mp4_paths.add(os.path.join(root, name).replace('boxes','videos').replace('.pt','.mp4'))
        elif name.endswith(".mp4"):
            video_mp4_paths.add(os.path.join(root, name))
video_mp4_paths = list(video_mp4_paths)

meta_infos = []
for video_mp4_path in video_mp4_paths:
    meta_infos.append({"video_path": video_mp4_path})

json.dump(meta_infos, open(f"./data/{args.meta_info_name}_meta.json", "w"))
