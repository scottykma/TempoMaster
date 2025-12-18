import math
import os


import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms.functional import center_crop
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from loguru import logger


from wan_dit import WanModel
from wan_dit import WanTextEncoder
from wan_dit import WanVideoVAE
from wan_dit import WanPrompter
from scheduler import FlowMatchScheduler
from .base import BasePipeline
from .guidance import FlowMatchingGuidance

def resize_and_crop(image, target_size):
    original_width, original_height = image.size
    target_width, target_height = target_size
    
    scale = max(target_width / original_width, target_height / original_height)
    
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    left = (new_width - target_width) / 2
    top = (new_height - target_height) / 2
    right = (new_width + target_width) / 2
    bottom = (new_height + target_height) / 2
    
    cropped_image = resized_image.crop((left, top, right, bottom))
    
    return cropped_image

def resize_and_crop_tensor(image, target_height, target_width):
    # image: b, t, c, h, w
    b, t, c, height, width = image.shape
    scale = max(target_width / width, target_height / height)
    
    new_height = int(height * scale)
    new_width = int(width * scale)

    resized_image = F.interpolate(
        image.reshape(-1, c, height, width),
        size=(new_height, new_width),
        mode='bilinear'
    ).reshape(b, t, c, new_height, new_width)

    cropped_image = center_crop(resized_image, (target_height, target_width))
    
    return cropped_image
 
class SequentialPipeline(BasePipeline):

    def __init__(
            self, 
            device="cuda:0", 
            torch_dtype=torch.float16, 
            tokenizer_path=None,
        ):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ['text_encoder', 'dit', 'vae']
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.patch_size = 2
        self.freqs = None

    @staticmethod
    def init_from_config(
            model_config,
            device="cuda:0",
            torch_dtype=torch.bfloat16, 
        ):
        pipe = SequentialPipeline(
            device=device, 
            torch_dtype=torch_dtype,
        )

        
        wan_video_text_encoder_path = model_config.get('wan_video_text_encoder_path', None)
        logger.info(f"Load TextEncoder from {wan_video_text_encoder_path}")
        wan_video_text_encoder = WanTextEncoder()
        wan_video_text_encoder.load_state_dict(
            torch.load(wan_video_text_encoder_path, weights_only=True, map_location='cpu')
        )
        pipe.text_encoder = wan_video_text_encoder.eval().to(torch_dtype)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(os.path.join(os.path.dirname(wan_video_text_encoder_path), "google/umt5-xxl"))
        
        
        wan_video_vae_path = model_config.get('wan_video_vae_path', None)
        logger.info(f"Load VAE from {wan_video_vae_path}")
        wan_video_vae = WanVideoVAE()
        wan_video_vae.model.load_state_dict(
            torch.load(wan_video_vae_path, weights_only=True, map_location='cpu')
        )
        pipe.vae = wan_video_vae.eval().to(torch_dtype)

        
        wan_video_dit_path = model_config.get('wan_video_dit_path', None)
        logger.info(f"Load DiT from {wan_video_dit_path}")
        wan_video_dit_config = model_config.get('wan_video_dit_config', None)

        dit = WanModel(
            **wan_video_dit_config
        )
        dit.load_state_dict(
            torch.load(wan_video_dit_path, weights_only=True, map_location='cpu')
        )
        pipe.freqs = dit.freqs
        pipe.dit = dit.eval().to(torch_dtype)
        
        logger.info("Pipeline Initiaized.")
        return pipe
    
    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive, device=self.device)
        return {"context": prompt_emb}
    
    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
  
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames

    def preparer_condition_video(self, frames, this_frame_interval, next_frame_interval):
        b, c, t, h, w = frames.shape
        ratio = this_frame_interval // next_frame_interval
        new_nframes = t * ratio
        
        n_chunks = max(4 // next_frame_interval, 1)
        nframes_per_chunk = (new_nframes // n_chunks + 2) // 4 * 4 + 1 # round it up

        new_frames = torch.zeros(b, c, new_nframes, h, w).to(frames.device, frames.dtype)
        masks = torch.zeros(b, new_nframes, h // 8, w // 8).to(frames.device, frames.dtype)

        for chunk_idx in range(n_chunks):
            new_frames[:, :, chunk_idx * nframes_per_chunk] = frames[:, :, chunk_idx * nframes_per_chunk // ratio]
            masks[:, chunk_idx * nframes_per_chunk] = 1

            if chunk_idx < n_chunks - 1:
                new_frames[:, :, (chunk_idx + 1) * nframes_per_chunk - 1] = frames[:, :, ((chunk_idx + 1) * nframes_per_chunk - 1) // ratio]
                masks[:, (chunk_idx + 1) * nframes_per_chunk - 1] = 1
            else:
                end_idx = min(n_chunks * nframes_per_chunk - 1, new_nframes - 1)
                new_frames[:, :, end_idx] = frames[:, :, end_idx // ratio]
                masks[:, end_idx] = 1

        anchors = []
        anchors.extend(list(range(t))) # maybe drop some frames 
        for idx in anchors:
            new_frames[:, :, idx * ratio] = frames[:, :, idx]
            masks[:, idx * ratio] = 1

        condition_frames = [] # List[b, c, t, h, w]
        condition_masks = []
       
        for chunk_idx in range(n_chunks):
            if chunk_idx == 0 or chunk_idx < n_chunks - 1:
                condition_frames.append(
                    new_frames[:, :, chunk_idx * nframes_per_chunk:(chunk_idx+1) * nframes_per_chunk]
                )
                condition_masks.append(
                    masks[:, chunk_idx * nframes_per_chunk:(chunk_idx+1) * nframes_per_chunk]
                )
            else:
                # keep the same shape for batch infer
                last_chunk_frames = torch.zeros_like(condition_frames[-1])
                last_chunk_mask = torch.zeros_like(condition_masks[-1])

                frames_left = min(new_nframes - chunk_idx * nframes_per_chunk, nframes_per_chunk)
                last_chunk_frames[:, :, :frames_left] = new_frames[:, :, chunk_idx * nframes_per_chunk:chunk_idx * nframes_per_chunk + frames_left]
                last_chunk_mask[:, :frames_left] = masks[:, chunk_idx * nframes_per_chunk:chunk_idx * nframes_per_chunk + frames_left]

                condition_frames.append(last_chunk_frames)
                condition_masks.append(last_chunk_mask)

        return torch.concat(condition_frames).to(dtype=self.torch_dtype, device=self.device), torch.concat(condition_masks).to(dtype=self.torch_dtype, device=self.device)

    def denoise_chunk(
            self, 
            latents,
            progress_bar_cmd,    
            frame_interval,
            width, height,
            prompt_emb_posi, prompt_emb_nega, image_emb,
            cfg_scale,
            enable_apg,
    ):
        b, c, f, h, w = latents.shape
        h, w = h // self.patch_size, w // self.patch_size

        freqs_batch = []
        for idx in range(b):
            this_frame_start = idx * (f * frame_interval)
            freqs = torch.cat([
                self.freqs[0][this_frame_start:this_frame_start + f * frame_interval: frame_interval].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1) 
            ] , dim=-1).reshape(f * h * w, 1, -1).to(latents.device)
            freqs_batch.append(freqs)
        freqs_batch = torch.stack(freqs_batch, dim=0) # (b, f*h*w, 1, -1)
        freqs_batch = torch.view_as_real(freqs_batch) # support dp 


        guidance = FlowMatchingGuidance(
            guidance_scale=cfg_scale,
            use_apg=enable_apg,
        )

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps, desc=f'Denoising: Frame Interval {frame_interval}, Resolution: {(width, height)}')):
                timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
                
                context = prompt_emb_posi['context']
                
                # Inference
                noise_pred_posi = self.dit(
                    latents, 
                    timestep=timestep, 
                    freqs=freqs_batch,
                    context=context.expand(b, -1, -1),
                    y=image_emb["y"]
                )

                if cfg_scale != 1.0:
                    noise_pred_nega = self.dit(
                    latents, 
                    timestep=timestep, 
                    freqs=freqs_batch,
                    context=prompt_emb_nega['context'].expand(b, -1, -1),
                    y=image_emb["y"]
                )
                    
                if cfg_scale != 1.0:
                    noise_pred = guidance(noise_pred_posi, noise_pred_nega, latents, timestep / 1000)
                else:
                    noise_pred = noise_pred_posi

                # Scheduler
                latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents, denoising_strength=1)
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        ref_images=None,
        denoising_strength=1.0,
        seed=0,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        progress_bar_cmd=tqdm,
        verbose=False,
        enable_apg=False,
        parrallel_config=dict(
            steps_per_stage=[50, 30],
            shift_per_stage=[5, 5],
            frame_intervals=[4, 1],
        ),
        **kwargs,
    ):
        torch.manual_seed(seed)

        assert 'steps_per_stage' in parrallel_config \
            and 'shift_per_stage' in parrallel_config \
                and 'frame_intervals' in parrallel_config, \
        "parrallel_config must contain 'steps_per_stage', 'shift_per_stage', and 'frame_intervals'"
        
        steps_per_stage = parrallel_config['steps_per_stage']
        shift_per_stage = parrallel_config['shift_per_stage']
        frame_intervals = parrallel_config['frame_intervals']

        # Parameter check
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Encode prompts
        self.enable_cpu_offload()
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        # resize
        target_area = width * height
        if ref_images:
            ref_images = [(int(frame_id), self.preprocess_image(Image.open(frame_path).convert('RGB'))) for frame_id, frame_path in ref_images.items()]
            original_width, original_height = ref_images[0][1].shape[-1], ref_images[0][1].shape[-2]
            ratio = original_height / original_width
            new_width, new_height = math.sqrt(target_area / ratio), math.sqrt(target_area * ratio)
            width = int(round(new_width / 16) * 16)
            height = int(round(new_height / 16) * 16)

        # Initialize noise
        noise = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height//8, width//8),
            seed=seed, 
            device=rand_device, 
            dtype=torch.float32
        )
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        latents = noise

        # Encode image
        if ref_images is not None:
            self.load_models_to_device(["vae"])
            b, c, t, h, w = latents.shape
            ref_video = torch.zeros(b, 4*t - 3, 3, 8 * h, 8 * w)
          
            for frame_id, ref_image in ref_images:
                ref_video[:, frame_id] = resize_and_crop_tensor(ref_image.unsqueeze(0), 8 * h, 8 * w)[0]

            ref_video = ref_video.to(dtype=self.torch_dtype, device=self.device).permute(0, 2, 1, 3, 4)
            
            ref_latents = self.vae.encode(
                ref_video, device=self.device, 
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
            ).to(dtype=self.torch_dtype, device=self.device)

            msk = torch.zeros(1, 4*t-3, h, w, device=self.device)
            for frame_id, _ in ref_images:
                msk[:, frame_id] = 1
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2).to(dtype=self.torch_dtype, device=self.device)
            y = torch.concat([msk, ref_latents], dim=1)

            image_emb = {}
            image_emb["y"] = y
        else:
            image_emb = {}
        
        # Denoise
        self.load_models_to_device(["dit", "vae"])

        history_frames = []

        frame_intervals = sorted(frame_intervals, reverse=True)

        for stage, frame_interval in enumerate(frame_intervals):
            new_chunks = []
            self.scheduler.set_timesteps(
                num_inference_steps=steps_per_stage[stage], 
                denoising_strength=denoising_strength, 
                shift=shift_per_stage[stage]
            )

            if stage == 0: 
                latents = self.denoise_chunk(
                    latents=noise,
                    progress_bar_cmd=progress_bar_cmd,
                    frame_interval=frame_interval,
                    width=width, height=height,
                    prompt_emb_posi=prompt_emb_posi, prompt_emb_nega=prompt_emb_nega, image_emb=image_emb, 
                    cfg_scale=cfg_scale,
                    enable_apg=enable_apg,
                )
                new_chunks.append(latents)
            else: 
                condition_chunks, masks = self.preparer_condition_video(
                    frames, 
                    frame_intervals[stage - 1], 
                    frame_intervals[stage],
                )
                b, c, t, h, w = condition_chunks.shape
                

                ref_latents = self.vae.encode(
                    condition_chunks, device=self.device, 
                    tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
                ).to(dtype=self.torch_dtype, device=self.device)
                masks = torch.concat([torch.repeat_interleave(masks[:, 0:1], repeats=4, dim=1), masks[:, 1:]], dim=1)
                masks = masks.view(b, masks.shape[1] // 4, 4, h//8, w//8)
                masks = masks.transpose(1, 2).to(dtype=self.torch_dtype, device=self.device)
                
                y = torch.concat([masks, ref_latents], dim=1)
                image_emb = {
                    'y': y
                }

                noise = self.generate_noise_like(
                    ref_latents, 
                    seed=seed, 
                    device=rand_device, 
                    dtype=torch.float32
                )
                noise = noise.to(dtype=self.torch_dtype, device=self.device)
                
                new_chunks = torch.chunk(
                    self.denoise_chunk(
                        latents=noise,
                        progress_bar_cmd=progress_bar_cmd,
                        frame_interval=frame_interval,
                        width=width, height=height,
                        prompt_emb_posi=prompt_emb_posi, prompt_emb_nega=prompt_emb_nega, image_emb=image_emb,
                        cfg_scale=cfg_scale,
                        enable_apg=enable_apg,
                    ),
                    chunks=b
                )

            # Decode current frames
            frame_list = []
            for _, latents in enumerate(new_chunks):
                chunk_frame = self.decode_video(latents, **tiler_kwargs)
                
                frame_list.append(
                    chunk_frame
                )

                
            frames = torch.concat(frame_list, dim=2)
            history_frames.append(self.tensor2video(frames[0]))
        
        self.load_models_to_device([])
        
        ret = [self.tensor2video(frames[0])]      
        if verbose:
            ret.append(history_frames)
        
        return ret
