import os
from typing import Dict, List, Tuple

import torch
from PIL import Image
from diffusers.utils import export_to_video

from pipelines import SequentialPipeline
import json


negative_prompt = "jittery, shaky, fliker, unstable, static, ultra slow motion, distorted, blurry details, many people in the background" + \
"worst quality, low quality, ugly, poorly drawn legs, poorly drawn face, three arms, " + \
"deformed, mutilated, disfigured, malformed limbs, fused fingers, three legs"

extra_prompt = ""
TASK_TYPE = 'i2v'
MODEL_DIR = 'long_video_sft_v2'
STEP = 18000
USE_EMA = True

model_name = 'diffusion_pytorch_model_ema.bin' if USE_EMA else 'diffusion_pytorch_model.bin'

DEFAULT_CONFIG = {
    "height": 480,
    "width": 832,
    "num_frames": 109, # 121 for 20s, 97 for 16s, ...
    "cfg_scale": 8, # could be a large scale for apg, change to 3-5 for cfg
    "tile": False,
    "save_fps": 24,
    "negative_prompt":negative_prompt,
    "use_apg": True,
    "verbose": False
} 

PARALLEL_CONFIG = dict(
    frame_intervals=[4, 2, 1],
    steps_per_stage=[50, 32, 32],
    shift_per_stage=[5, 3, 3],
)


MODEL_CONFIG = dict(
    wan_video_text_encoder_path="encoders/models_t5_umt5-xxl-enc-bf16.pth",
    wan_video_vae_path='encoders/Wan2.1_VAE.pth',
    wan_video_dit_path=f'/nvfile-heatstorage/AIGC_H100/myk/models/{MODEL_DIR}/teletron_step{STEP}/{model_name}',
    wan_video_dit_config=dict(
            has_image_input=False, # t2v:False i2v:True i2v Wan2.2:False
            patch_size=[1, 2, 2],
            in_dim=36, # t2v:16 i2v:36
            dim=5120, # 1.3B:1536 10B:5120 14B:5120
            ffn_dim=13824, # 1.3B:8960 10B:13824 14B:13824
            freq_dim=256,
            text_dim=4096,
            out_dim=16,
            num_heads=40, # 1.3B:12 10B:40 14B:40
            num_layers=40, # 1.3B:30 10B:30 14B:40
            eps=1e-6,
            has_image_pos_emb=False, 
        ),
)

SAVEDIR = f"results/{MODEL_DIR}/step_{STEP}"+('_ema' if USE_EMA else '')
GPU_IDS = [0,1,2,3,4,5,6,7]

class InferenceConfig:
    def __init__(
        self,
        prompt: str,
        ref_images: Dict[int, str],
        height: int = DEFAULT_CONFIG["height"],
        width: int = DEFAULT_CONFIG["width"],
        num_frames: int = DEFAULT_CONFIG["num_frames"],
        cfg_scale: float = DEFAULT_CONFIG["cfg_scale"],
        tile: bool = DEFAULT_CONFIG["tile"],
        save_fps: int = DEFAULT_CONFIG["save_fps"],
        negative_prompt: str = DEFAULT_CONFIG["negative_prompt"],
        **kwargs
    ):
        self.prompt = prompt
        self.ref_images = ref_images
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.cfg_scale = cfg_scale
        self.tile = tile
        self.save_fps = save_fps
        self.negative_prompt = negative_prompt

def generate_video_filename(
    result_name: str, 
    config: InferenceConfig
) -> str:
    
    """生成视频文件名"""
    width, height = config.width, config.height
    base_name = f"{result_name}_{height}x{width}_f{config.num_frames}x{PARALLEL_CONFIG['frame_intervals'][0]}_g{config.cfg_scale}_step{'_'.join([str(step) for step in PARALLEL_CONFIG['steps_per_stage']])}"
    return f"{base_name}.mp4"

def prepare_reference_images(config: InferenceConfig) -> List[Image.Image]:
    ref_images = []
    is_vertical = []
    for _, img_path in config.ref_images.items():
        original_img = Image.open(img_path).convert("RGB")
        ref_images.append(original_img)
        is_vertical.append(original_img.height > original_img.width)
    return ref_images, is_vertical

def inference_worker(
    rank: int,
    inference_configs: List[Tuple[str, InferenceConfig]],
):

    pipe = SequentialPipeline.init_from_config(
        MODEL_CONFIG, 
        torch_dtype=torch.bfloat16, 
        device=f'cuda:{rank}'
    )

    
    for test_name, config in inference_configs[rank::2]:
        try:
            import time
            tik = time.time()
            save_dir = os.path.join('results', SAVEDIR)
            os.makedirs(save_dir, exist_ok=True)

            save_name = generate_video_filename(test_name, config)
            save_path = os.path.join(save_dir, save_name)

            if TASK_TYPE == 't2v':
                ref_images = {}
            else:
                ref_images = config.ref_images
           
            
            result = pipe(
                prompt=extra_prompt + config.prompt,
                negative_prompt=config.negative_prompt,
                ref_images=ref_images,
                height=config.height,
                width=config.width,
                num_frames=config.num_frames,
                cfg_scale=config.cfg_scale,
                seed=0,
                tiled=config.tile,
                verbose=DEFAULT_CONFIG["verbose"],
                enable_apg=DEFAULT_CONFIG["use_apg"],
                parrallel_config=PARALLEL_CONFIG
            )
           
            if len(result) == 2:
                [output, history_frames] = result
            else:
                output = result[0]
                history_frames = []
            export_to_video(output, str(save_path), fps=config.save_fps, quality=6)
            
            frame_intervals = PARALLEL_CONFIG["frame_intervals"]
            for i, frames in enumerate(history_frames):
                if frame_intervals[i] == 1:
                    continue
                save_name = generate_video_filename(config.prompt, config)
                save_path = os.path.join(save_dir, save_name)
                export_to_video(
                    frames, 
                    str(save_path).replace(save_name, test_name) + f'_verbose_fps{24 // frame_intervals[i]}.mp4', 
                    fps=24 // frame_intervals[i], quality=8
                )

            tok = time.time()
            print(f"[Rank {rank}] Time Cost: {(tok - tik)} seconds")

        except Exception as e:
            import traceback
            traceback.print_exc()
    
def run_inference_pipeline(
    prompt_configs: Dict[str, Dict[str, str]],
):
    inference_config_list = []
    for test_name, config_data in prompt_configs.items():
        inference_config = InferenceConfig(
            prompt=config_data["prompt"],
            ref_images=config_data["ref_images"],
            **{k: v for k, v in DEFAULT_CONFIG.items()},
        )
        inference_config_list.append((test_name, inference_config))


    import torch.multiprocessing as mp
    from functools import partial
    mp.spawn(
        partial(
            inference_worker,
            inference_configs=inference_config_list,
        ),
        nprocs=8,
        join=True,
    )


if __name__ == "__main__":
    try:
        with open('benchmarks/inference_demos.json', 'r') as f:
            data = json.load(f)
        run_inference_pipeline(
            prompt_configs=data,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
