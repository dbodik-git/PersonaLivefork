import sys
sys.path.append('./src/stylegan2')

import argparse
import logging
import math
import os
import os.path as osp
import random
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from torchvision import transforms

import cv2
import lpips
import diffusers
from einops import rearrange
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, AutoencoderKLTemporalDecoder, AutoencoderTiny
from src.scheduler.scheduler_ddim import DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection
from decord import VideoReader
from src.dataset.portrait_video import PortraitVideoDataset
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid import Pose2VideoPipeline
from src.utils.util import (
    delete_additional_ckpt,
    import_filename,
    read_frames,
    save_videos_grid,
    seed_everything,
    draw_keypoints,
)
from src.dataset.utils import scale_bb
from src.models.motion_encoder.encoder import MotEncoder
from src.models.motion_module import zero_module
warnings.filterwarnings("ignore")
import mediapipe as mp
from src.liveportrait.motion_extractor import MotionExtractor
from src.models.pose_guider import PoseGuider
import gc
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")
from torchvision.utils import save_image
from src.stylegan2.discriminator import Discriminator as StyleGAN2Discriminator

logger = get_logger(__name__, log_level="INFO")

class Net(nn.Module):
    def __init__(
        self,
        denoising_unet: UNet3DConditionModel,
        reference_control_reader,
    ):
        super().__init__()
        self.denoising_unet = denoising_unet
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        clip_image_embeds,
        motion,
        pose,
    ):
        motion_emb = motion

        clip_image_embeds = [clip_image_embeds, motion_emb]

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=clip_image_embeds,
            pose_cond_fea=pose,
        ).sample

        return model_pred

def get_x0_from_eps(ddim, sample: torch.FloatTensor, model_output: torch.FloatTensor, timesteps: torch.IntTensor):
    video_length = sample.shape[2]
    sample = rearrange(sample, 'b c f h w -> (b f) c h w')
    model_output = rearrange(model_output, 'b c f h w -> (b f) c h w')
    alpha_prod = ddim.alphas_cumprod.cuda()
    alpha_prod_t = alpha_prod[timesteps]
    while len(alpha_prod_t.shape) < len(sample.shape):
        alpha_prod_t = alpha_prod_t.unsqueeze(-1)

    pred_original_sample = torch.sqrt(1./alpha_prod_t) * sample - torch.sqrt(1. / alpha_prod_t - 1) * model_output

    pred_original_sample = rearrange(pred_original_sample, '(b f) c h w -> b c f h w', f=video_length)

    return pred_original_sample

def crop_face(image_pil, face_mesh):
    image = np.array(image_pil)
    h, w = image.shape[:2]
    results = face_mesh.process(image)
    face_landmarks = results.multi_face_landmarks[0]
    coords = [(int(l.x * w), int(l.y * h)) for l in face_landmarks.landmark]
    xs, ys = zip(*coords)
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    face_box = (x1, y1, x2, y2)

    left, top, right, bot = scale_bb(face_box, scale=1.1, size=image.shape[:2])

    face_patch = image[int(top) : int(bot), int(left) : int(right)]

    return face_box, face_patch

def decode_latents(vae: AutoencoderKL, latents, decode_chunk_size=1):
    latents = latents.to(vae.dtype)
    video_length = latents.shape[2]
    latents = 1 / 0.18215 * latents
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    # video = self.vae.decode(latents).sample
    video = []
    for frame_idx in range(0, latents.shape[0], decode_chunk_size):
        video.append(vae.decode(latents[frame_idx : frame_idx + decode_chunk_size]).sample)
    video = torch.cat(video)
    video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
    # video = (video / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    return video

def log_validation(
    vae,
    image_enc,
    net,
    scheduler,
    accelerator,
    width,
    height,
    face_mesh,
    reference_unet,
    pose_encoder,
    pose_guider,
    motion_encoder,
    timesteps_list,
    clip_length=100,
    generator=None,
):
    logger.info("Running validation... ")

    ori_net = accelerator.unwrap_model(net)
    reference_unet = reference_unet
    denoising_unet = ori_net.denoising_unet
    motion_enc = motion_encoder
    pose_gui = pose_guider
    pose_enc = pose_encoder

    # generator = torch.manual_seed(42)
    generator = torch.Generator(device='cuda').manual_seed(42)
    # cast unet dtype

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        motion_encoder=motion_enc,
        pose_encoder=pose_enc,
        pose_guider=pose_gui,
        scheduler=scheduler,
    )
    pipe = pipe.to(accelerator.device)

    ref_image_paths = [
        "path/to/ref_img_1.png",
        "path/to/ref_img_2.png",
    ]
    videos_path = [
        "path/to/ref_vid_1.png",
        "path/to/ref_vid_2.png",
    ]

    pose_transform = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()]
    )
    results = []
    for idy, pose_video_path in enumerate(videos_path):
        pose_images = read_frames(pose_video_path)

        boxes_path = pose_video_path.replace("videos", "boxes").replace(".mp4", ".pt")
        boxes = torch.load(boxes_path)

        pose_face = []
        ori_pose_images = []
        for idx_control, pose_image_pil in enumerate(pose_images[:clip_length]):
            face_bbox = boxes[idx_control]["face"]
            ori_pose_images.append(pose_image_pil)
            pose_image = np.array(pose_image_pil)
            left, top, right, bot = scale_bb(face_bbox, scale=1.1, size=pose_image.shape[:2])
            pose_image = pose_image[int(top) : int(bot), int(left) : int(right)]
            pose_image_pil = Image.fromarray(pose_image).convert("RGB")
            pose_face.append(pose_image_pil)

        pose_tensor_list = []
        ori_pose_tensor_list = []

        for idx_control, pose_image_pil in enumerate(ori_pose_images):
            pose_tensor_list.append(pose_transform(pose_face[idx_control]))
            ori_pose_tensor_list.append(pose_transform(pose_image_pil))
        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1).unsqueeze(0)

        ori_pose_tensor = torch.stack(ori_pose_tensor_list, dim=0)  # (f, c, h, w)
        ori_pose_tensor = ori_pose_tensor.transpose(0, 1).unsqueeze(0)

        for idx, ref_image_path in enumerate(ref_image_paths):
            ref_image_pil = Image.open(ref_image_path).convert("RGB")
            ref_bbox, ref_patch = crop_face(ref_image_pil, face_mesh)
            ref_face = Image.fromarray(ref_patch)
            
            ref_tensor = pose_transform(ref_image_pil).unsqueeze(0).unsqueeze(2).expand_as(pose_tensor)

            ref_name = f'{idx}'
            pose_name = f'{idy}'

            with torch.cuda.amp.autocast():
                gen_video = pipe(
                    ori_pose_images,
                    ref_image_pil,
                    pose_face,
                    ref_face,
                    width,
                    height,
                    clip_length,
                    4,
                    1.0,
                    timesteps_list=timesteps_list,
                    generator=generator,
                    temporal_window_size = 4,
                    temporal_adaptive_step = 4,
                    temporal_kv_cache=False,
                ).videos
            
            # Concat it with pose tensor
            video = torch.cat([ref_tensor, pose_tensor, ori_pose_tensor, gen_video], dim=0)

            results.append({"name": f"{ref_name}_{pose_name}", "vid": video})

    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    return results


def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, static_graph=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    if accelerator.is_main_process and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    inference_config_path = "./configs/inference/inference_stage3.yaml"
    infer_config = OmegaConf.load(inference_config_path)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(
        infer_config.noise_scheduler_kwargs
    )

    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )
    net_lpips = lpips.LPIPS(net='vgg').cuda()

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")

    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda", dtype=weight_dtype)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        '',
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            infer_config.unet_additional_kwargs
        ),
    ).to(device="cuda")

    motion_encoder = MotEncoder().to(device="cuda", dtype=weight_dtype).eval()
    motion_encoder.load_state_dict(
        torch.load(cfg.motion_encoder_path, map_location="cpu"), strict=True
    )

    pose_guider = PoseGuider().to(device="cuda", dtype=weight_dtype)
    pose_guider.load_state_dict(torch.load(cfg.pose_guider_path, map_location='cpu'), strict=True)

    pose_encoder = MotionExtractor(num_kp=21).to(device="cuda").eval()
    pose_encoder.load_state_dict(torch.load(cfg.pose_encoder_path, map_location='cpu'), strict=False)

    denoising_unet.load_state_dict(
        torch.load(cfg.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    denoising_unet.load_state_dict(
        torch.load(cfg.temporal_module_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(cfg.reference_unet_path, map_location="cpu"),
    )

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    discriminator = StyleGAN2Discriminator()
    discriminator = discriminator.to(device="cuda")
    discriminator.load_state_dict(
        torch.load(cfg.discriminator_path, map_location="cpu"), strict=True
    )

    # Freeze
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    net_lpips.requires_grad_(False)
    pose_encoder.requires_grad_(False)

    pose_guider.requires_grad_(False)
    motion_encoder.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    reference_unet.requires_grad_(False)

    discriminator.requires_grad_(True)
    for name, module in denoising_unet.named_modules():
        if "temporal_modules" in name:
            for params in module.parameters():
                params.requires_grad = True

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    net = Net(
        denoising_unet,
        reference_control_reader,
    )

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()
        
    # Ensure that no out-of-memory (OOM) error occurs
    vae.decoder.gradient_checkpointing = True
    vae.decoder.training = True

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))

    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    dis_optimizer = optimizer_cls(
        discriminator.parameters(),
        lr=learning_rate * 2,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    train_dataset = PortraitVideoDataset(
        img_size=(cfg.data.train_width, cfg.data.train_height),
        img_scale=(0.9, 1.0),
        data_meta_paths=cfg.data.meta_paths,
        n_sample_frames=cfg.data.n_sample_frames,
        sample_rate=cfg.data.sample_rate,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=8, drop_last=True
    )

    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        discriminator,
        dis_optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        discriminator,
        dis_optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            cfg.exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )
        # dump config file
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

    # Train!
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = save_dir
        # Get the most recent checkpoint
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    temporal_window_size = cfg.temporal_window_size
    temporal_adaptive_step = cfg.temporal_adaptive_step

    windows = cfg.data.n_sample_frames // temporal_window_size

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        train_dis_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Convert videos to latent space
            pixel_values = batch["img"].to(weight_dtype) # b,c,h,w
            video_length = pixel_values.shape[1]

            tgt_face = batch["tgt_face"]
            tgt_pose = batch["tgt_pose"]
            
            tgt_face = rearrange(tgt_face, "b f c h w -> (b f) c h w")
            tgt_pose = rearrange(tgt_pose, "b f c h w -> (b f) c h w")

            face_mask = batch['face_mask']
            local_mask = batch['local_mask']

            face_mask = rearrange(face_mask, "b f c h w -> b c f h w")
            local_mask = rearrange(local_mask, "b f c h w -> b c f h w")

            facial_mask_pix = face_mask.repeat(1,3,1,1,1) + local_mask.repeat(1,3,1,1,1)
            
            pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
            with torch.no_grad():
                latents = []
                for i in range(0, pixel_values.shape[0], cfg.data.max_batch_frames):
                    batch_latents = vae.encode(
                        pixel_values[i : i + cfg.data.max_batch_frames]
                    ).latent_dist.sample()
                    latents.append(batch_latents)
                latents = torch.cat(latents, dim=0)
                latents = latents * 0.18215 # (b f) c h w
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)

            pixel_values = rearrange(pixel_values, "(b f) c h w -> b c f h w", f=video_length)
            noise = torch.randn_like(latents)

            bsz = latents.shape[0] # (b f)
            timesteps_list = cfg.timesteps_list
            timesteps = torch.tensor(timesteps_list[::-1], device=latents.device)
            timesteps = timesteps.repeat_interleave(temporal_window_size, dim=0).long()

            uncond_fwd = random.random() < cfg.uncond_ratio
            clip_image_list = []
            ref_image_list = []
            for batch_idx, (ref_img, clip_img) in enumerate(
                zip(
                    batch["ref_img"],
                    batch["clip_image"],
                )
            ):
                clip_image_list.append(clip_img)
                ref_image_list.append(ref_img)

            with torch.no_grad():
                ref_img = torch.stack(ref_image_list, dim=0).to(
                    dtype=vae.dtype, device=vae.device
                )
                ref_image_latents = vae.encode(
                    ref_img
                ).latent_dist.sample()  # (bs, d, 64, 64)
                ref_image_latents = ref_image_latents * 0.18215

                clip_img = torch.stack(clip_image_list, dim=0).to(
                    dtype=image_enc.dtype, device=image_enc.device
                )
                clip_image_embeds = image_enc(
                    clip_img.to("cuda", dtype=weight_dtype)
                ).image_embeds
                image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)

            init_frames = (temporal_adaptive_step - 1) * temporal_window_size
            # add noise
            noisy_latents = train_noise_scheduler.add_noise(
                latents[:,:,:init_frames], noise[:,:,:init_frames], timesteps[:init_frames]
            ) # (b f) c h w

            noisy_latents_all = torch.cat([noisy_latents, noise[:,:,init_frames:]], dim=2)

            mot_flag = random.random() < 0.1
            pose_flag = random.random() < 0.1
            reference_control_reader.clear()
            reference_control_writer.clear()
            
            with torch.no_grad():
                keypoints = pose_encoder(tgt_pose)
                tgt_pose = draw_keypoints(keypoints).to('cuda')
                tgt_pose = rearrange(tgt_pose, '(b f) c h w -> b c f h w', b=bsz)
                pose_emb = pose_guider(tgt_pose.to(pose_guider.dtype))
                motion_emb = motion_encoder(tgt_face)

                if mot_flag:
                    motion_emb = torch.zeros_like(motion_emb)
                if pose_flag:
                    pose_emb = torch.zeros_like(pose_emb)
                
                # reference feature
                if not uncond_fwd:
                    reference_unet(
                        ref_image_latents,
                        torch.zeros_like(timesteps[:bsz]),
                        encoder_hidden_states=image_prompt_embeds,
                        return_dict=False,
                    )
                    reference_control_reader.update(reference_control_writer, drop_ratio=0.3)
            
            for i in range(windows - temporal_adaptive_step + 1):
                l = i * temporal_window_size
                r = (i + temporal_adaptive_step) * temporal_window_size

                noisy_latents = noisy_latents_all[:,:,l:r,:,:].detach()
                motion = motion_emb[l:r]
                pose = pose_emb[:,:,l:r]

                if i % (temporal_adaptive_step - 1) != 0:
                    with torch.no_grad():
                        model_pred = net(
                            noisy_latents,
                            timesteps,
                            image_prompt_embeds,
                            motion,
                            pose,
                        )

                        latents_pred = get_x0_from_eps(train_noise_scheduler, noisy_latents, model_pred, timesteps)
                        noise = torch.randn_like(model_pred[:,:,temporal_window_size:])
                        noisy_latents = train_noise_scheduler.add_noise(
                            latents_pred[:,:,temporal_window_size:], noise, timesteps[:-temporal_window_size]
                        )
                        mid_latents = torch.cat([latents_pred[:,:,:temporal_window_size], noisy_latents], dim=2)
                        noisy_latents_all[:,:,l:r,:,:] = mid_latents.detach()
                else:
                    with accelerator.accumulate(net):
                        model_pred = net(
                            noisy_latents,
                            timesteps,
                            image_prompt_embeds,
                            motion,
                            pose,
                        )

                        discriminator.requires_grad_(False)

                        latents_pred = get_x0_from_eps(train_noise_scheduler, noisy_latents, model_pred, timesteps)
                        image_pred = decode_latents(vae, latents_pred)

                        pixel_values_tgt = pixel_values[:, :, l:r,:,:]

                        if epoch == 0 and step == 0:
                            if accelerator.is_main_process:
                                save_image((ref_img[:1] + 1) / 2, "image_ref.png")
                                save_image((pixel_values_tgt[:,:,0] + 1) / 2, "image_tgt.png")
                                save_image((image_pred[:,:,0] + 1) / 2, "image_pred.png")
                        facial_mask = facial_mask_pix[:, :, l:r, :, :]
                        loss = F.mse_loss(image_pred.float(), pixel_values_tgt.float(), reduction="mean")
                        loss += F.mse_loss(image_pred.float() * facial_mask, pixel_values_tgt.float() * facial_mask, reduction="mean")
                        image_pred = rearrange(image_pred, 'b c f h w -> (b f) c h w')
                        pixel_values_tgt = rearrange(pixel_values_tgt, 'b c f h w -> (b f) c h w')
                        loss_lpips = net_lpips(image_pred.float(), pixel_values_tgt.float()).mean() # 输入应该是[-1,1]
                        
                        adv_loss = discriminator(image_pred.float(), pixel_values_tgt.float(), timesteps)
                        loss = loss + loss_lpips * 2.0 + adv_loss * 0.05
                        
                        noise = torch.randn_like(model_pred[:,:,temporal_window_size:])
                        noisy_latents = train_noise_scheduler.add_noise(
                            latents_pred[:,:,temporal_window_size:], noise, timesteps[:-temporal_window_size]
                        )
                        mid_latents = torch.cat([latents_pred[:,:,:temporal_window_size], noisy_latents], dim=2)
                        noisy_latents_all[:,:,l:r,:,:] = mid_latents

                        # Gather the losses across all processes for logging (if we use distributed training).
                        avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                        train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                        # Backpropagate
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                trainable_params,
                                cfg.solver.max_grad_norm,
                            )
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                        if global_step > 50: # This value may not be optimal
                            discriminator.requires_grad_(True)
                            dis_loss = discriminator(image_pred.float().detach(), pixel_values_tgt.float(), timesteps, gen=False)
                            avg_loss = accelerator.gather(dis_loss.repeat(cfg.data.train_bs)).mean()
                            train_dis_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                            accelerator.backward(dis_loss)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(
                                    discriminator.parameters(),
                                    cfg.solver.max_grad_norm,
                                )
                            dis_optimizer.step()
                            dis_optimizer.zero_grad(set_to_none=True)
                        else:
                            dis_loss = torch.tensor(0.0)

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "train_dis_loss": train_dis_loss}, step=global_step)
                train_loss = 0.0
                train_dis_loss = 0.0
                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                        delete_additional_ckpt(save_dir, 1)
                        accelerator.save_state(save_path)

                if global_step % cfg.val.validation_steps == 0 or global_step == 1:
                    if accelerator.is_main_process:
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(cfg.seed)

                        sample_dicts = log_validation(
                            vae=vae,
                            image_enc=image_enc,
                            net=net,
                            scheduler=val_noise_scheduler,
                            accelerator=accelerator,
                            width=cfg.data.train_width,
                            height=cfg.data.train_height,
                            face_mesh=face_mesh,
                            reference_unet=reference_unet,
                            generator=generator,
                            pose_encoder=pose_encoder,
                            pose_guider=pose_guider,
                            motion_encoder=motion_encoder,
                            timesteps_list=timesteps_list,
                        )

                        reference_control_writer = ReferenceAttentionControl(
                            reference_unet,
                            do_classifier_free_guidance=False,
                            mode="write",
                            fusion_blocks="full",
                            )
                        reference_control_reader = ReferenceAttentionControl(
                            denoising_unet,
                            do_classifier_free_guidance=False,
                            mode="read",
                            fusion_blocks="full",
                            )

                        for sample_id, sample_dict in enumerate(sample_dicts):
                            sample_name = sample_dict["name"]
                            vid = sample_dict["vid"]
                            with TemporaryDirectory() as temp_dir:
                                out_file = f"{temp_dir}/{global_step:06d}-{sample_name}.mp4"
                                save_videos_grid(vid, out_file, n_rows=4, fps=25)
                                mlflow.log_artifact(out_file)

            logs = {
                "loss": loss.detach().item(),
                "dis": dis_loss.detach().item(),
                "adv": adv_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            # save model after each epoch
            if global_step % cfg.checkpointing_steps == 0 and accelerator.is_main_process:
                unwrap_net = accelerator.unwrap_model(net)
                unwrap_discriminator = accelerator.unwrap_model(discriminator)
                save_checkpoint(
                    unwrap_net.denoising_unet,
                    save_dir,
                    "temporal_module",
                    global_step,
                    total_limit=2,
                )
                save_checkpoint_dis(
                    unwrap_discriminator,
                    save_dir,
                    "discriminator",
                    global_step,
                    total_limit=2,
                )
            
            if global_step >= cfg.solver.max_train_steps:
                break
    
    unwrap_net = accelerator.unwrap_model(net)
    unwrap_discriminator = accelerator.unwrap_model(discriminator)
    save_checkpoint(
        unwrap_net.denoising_unet,
        save_dir,
        "temporal_module",
        global_step,
        total_limit=2,
    )
    save_checkpoint_dis(
        unwrap_discriminator,
        save_dir,
        "discriminator",
        global_step,
        total_limit=2,
    )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    mm_state_dict = OrderedDict()
    state_dict = model.state_dict()
    for key in state_dict:
        if "temporal_modules" in key:
            mm_state_dict[key] = state_dict[key]

    torch.save(mm_state_dict, save_path)

def save_checkpoint_dis(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/personalive_stage3.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)
