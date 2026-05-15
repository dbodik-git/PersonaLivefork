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
from diffusers import AutoencoderKL, AutoencoderKLTemporalDecoder
from src.scheduler.scheduler_ddim import DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection
from decord import VideoReader
from src.dataset.portrait_image import PortraitImageDataset
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2img import Pose2ImagePipeline
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

logger = get_logger(__name__, log_level="INFO")

class Net(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        motion_enc: MotEncoder,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

        self.motion_encoder = motion_enc
        self.pose_guider = pose_guider

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        tgt_face,
        tgt_pose,
        uncond_fwd: bool = False,
    ):  
        pose_emb = self.pose_guider(tgt_pose)
        motion_emb = self.motion_encoder(tgt_face)

        if not uncond_fwd:
            self.reference_unet(
                ref_image_latents,
                torch.zeros_like(timesteps[:noisy_latents.shape[0]]),
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer, drop_ratio=0.3)
        
        if random.random() < 0.1:
            pose_emb = torch.zeros_like(pose_emb)

        if random.random() < 0.1:
            motion_emb = torch.zeros_like(motion_emb)

        clip_image_embeds = [clip_image_embeds, motion_emb]

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=clip_image_embeds,
            pose_cond_fea=pose_emb,
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

def log_validation(
    vae,
    image_enc,
    net,
    scheduler,
    accelerator,
    width,
    height,
    face_mesh,
    pose_encoder,
    clip_length=100,
    generator=None,
):
    logger.info("Running validation... ")

    ori_net = accelerator.unwrap_model(net)
    reference_unet = ori_net.reference_unet
    denoising_unet = ori_net.denoising_unet
    motion_enc = ori_net.motion_encoder
    pose_gui = ori_net.pose_guider
    pose_enc = pose_encoder

    motion_enc.eval()

    # generator = torch.manual_seed(42)
    generator = torch.Generator(device='cuda').manual_seed(42)
    # cast unet dtype

    pipe = Pose2ImagePipeline(
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
                    20,
                    2.5,
                    generator=generator,
                ).videos
            
            # Concat it with pose tensor
            video = torch.cat([ref_tensor, pose_tensor, ori_pose_tensor, gen_video], dim=0)

            results.append({"name": f"{ref_name}_{pose_name}", "vid": video})

    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    motion_enc.train()
    return results


def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
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

    inference_config_path = "./configs/inference/inference_stage1&2.yaml"
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

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")

    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda")
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            infer_config.unet_additional_kwargs
        ),
    ).to(device="cuda")

    motion_encoder = MotEncoder().to(device="cuda")
    motion_encoder_params = torch.load(cfg.motion_encoder_path, map_location="cpu")
    old_weight = motion_encoder_params['final_proj.weight']
    new_weight = motion_encoder.final_proj.weight.data
    new_weight = old_weight[:, :new_weight.shape[1]] 
    motion_encoder_params['final_proj.weight'] = new_weight

    motion_encoder.load_state_dict(
        motion_encoder_params, strict=False
    )

    pose_guider = PoseGuider().to(device="cuda")
    pose_guider.load_state_dict(torch.load(cfg.pose_guider_path, map_location='cpu'), strict=False)

    pose_encoder = MotionExtractor(num_kp=21).to(device="cuda").eval()
    pose_encoder.load_state_dict(torch.load(cfg.pose_encoder_path, map_location='cpu'), strict=False)

    denoising_unet.load_state_dict(
        torch.load(cfg.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(cfg.reference_unet_path, map_location="cpu"),
    )

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    # Freeze
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    pose_encoder.requires_grad_(False)

    pose_guider.requires_grad_(True)
    motion_encoder.requires_grad_(True)
    denoising_unet.requires_grad_(True)
    reference_unet.requires_grad_(False)
    
    for name, param in reference_unet.named_parameters():
        if "up_blocks.3" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

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
        reference_unet,
        denoising_unet,
        motion_encoder,
        pose_guider,
        reference_control_writer,
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

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    train_dataset = PortraitImageDataset(
        img_size=(cfg.data.train_width, cfg.data.train_height),
        img_scale=(0.9, 1.0),
        data_meta_paths=cfg.data.meta_paths,
        sample_margin=cfg.data.sample_margin,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=8, drop_last=True
    )

    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
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

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(net):
                # Convert videos to latent space
                pixel_values = batch["img"].to(weight_dtype) # b,c,h,w
                tgt_face = batch["tgt_face"]
                tgt_pose = batch["tgt_pose"]

                face_mask = batch['face_mask']
                local_mask = batch['local_mask']
                
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents.unsqueeze(2)
                    latents = latents * 0.18215 # (b f) c h w

                    keypoints = pose_encoder(tgt_pose)
                    tgt_pose = draw_keypoints(keypoints).unsqueeze(2).to('cuda')
                    
                    if epoch == 0 and step == 0:
                         if accelerator.is_main_process:
                            save_image(tgt_pose[:1,:,0], "image_kps.png")
                            save_image((pixel_values[:1] + 1) / 2, "image_tgt.png")

                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0.0:
                    noise += cfg.noise_offset * torch.randn(
                        (noise.shape[0], noise.shape[1], 1, 1, 1),
                        device=noise.device,
                    )

                bsz = latents.shape[0] # (b f)
                # Sample a random timestep for each frame
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                facial_mask = F.interpolate(face_mask, size=(64,64), mode='nearest').unsqueeze(2).repeat(1,4,1,1,1) + F.interpolate(local_mask, size=(64,64), mode='nearest').unsqueeze(2).repeat(1,4,1,1,1)

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

                # add noise
                noisy_latents = train_noise_scheduler.add_noise(
                    latents, noise, timesteps
                ) # (b f) c h w

                reference_control_reader.clear()
                reference_control_writer.clear()

                model_pred = net(
                    noisy_latents,
                    timesteps,
                    ref_image_latents,
                    image_prompt_embeds,
                    tgt_face,
                    tgt_pose,
                    uncond_fwd,
                )
                
                latents_pred = get_x0_from_eps(train_noise_scheduler, noisy_latents, model_pred, timesteps)

                loss = F.mse_loss(latents_pred.float(), latents.float(), reduction="mean")
                loss_local = F.mse_loss(latents_pred.float() * facial_mask, latents.float() * facial_mask, reduction="mean")
                loss = loss + loss_local

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

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
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
                            pose_encoder=pose_encoder,
                            generator=generator,
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
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break

        # save model after each epoch
        if (
            epoch + 1
        ) % cfg.save_model_epoch_interval == 0 and accelerator.is_main_process:
            unwrap_net = accelerator.unwrap_model(net)
            save_checkpoint(
                unwrap_net.denoising_unet,
                save_dir,
                "denoising_unet",
                global_step,
                total_limit=2,
            )
            save_checkpoint(
                unwrap_net.reference_unet,
                save_dir,
                "reference_unet",
                global_step,
                total_limit=2,
            )
            save_checkpoint(
                unwrap_net.motion_encoder,
                save_dir,
                "motion_encoder",
                global_step,
                total_limit=2,
            )
            save_checkpoint(
                unwrap_net.pose_guider,
                save_dir,
                "pose_guider",
                global_step,
                total_limit=2,
            )
    
    unwrap_net = accelerator.unwrap_model(net)
    save_checkpoint(
        unwrap_net.denoising_unet,
        save_dir,
        "denoising_unet",
        global_step,
        total_limit=2,
    )
    save_checkpoint(
        unwrap_net.reference_unet,
        save_dir,
        "reference_unet",
        global_step,
        total_limit=2,
    )
    save_checkpoint(
        unwrap_net.motion_encoder,
        save_dir,
        "motion_encoder",
        global_step,
        total_limit=2,
    )
    save_checkpoint(
        unwrap_net.pose_guider,
        save_dir,
        "pose_guider",
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

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/personalive_stage1.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)
