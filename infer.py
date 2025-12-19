import os
from typing import Dict, List, Tuple
from argparse import ArgumentParser
import json

import torch
from PIL import Image
from diffusers.utils import export_to_video

from pipelines import SequentialPipeline

class InferenceConfig:
    def __init__(
        self,
        prompt: str,
        ref_images: Dict[int, str],
        height: int = 480,
        width: int = 832,
        num_frames: int = 109,
        cfg_scale: float = 8,
        tile: bool = False,
        save_fps: int = 24,
        negative_prompt: str = "",
        verbose: bool = False,
        use_apg: bool = True,
        parallel_configs: dict = {},
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
        self.verbose = verbose
        self.use_apg = use_apg
        self.parallel_configs = parallel_configs



def generate_video_filename(
    result_name: str, 
    config: InferenceConfig
) -> str:
    
    """生成视频文件名"""
    width, height = config.width, config.height
    base_name = f"{result_name}_{height}x{width}_f{config.num_frames}_g{config.cfg_scale}"
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
    args, 
    inference_case_list: List[Tuple[str, InferenceConfig]],
    config
):

    pipe = SequentialPipeline.init_from_config(
        model_config=config["model_configs"], 
        torch_dtype=torch.bfloat16, 
        device=f'cuda:{rank}'
    )

    
    for test_name, test_config in inference_case_list[rank::args.world_size]:
        try:
            import time
            tik = time.time()
            save_dir = args.output_dir
            os.makedirs(save_dir, exist_ok=True)

            save_name = generate_video_filename(test_name, test_config)
            save_path = os.path.join(save_dir, save_name)   
            ref_images = test_config.ref_images
           
            
            result = pipe(
                prompt=test_config.prompt,
                negative_prompt=test_config.negative_prompt,
                ref_images=ref_images,
                height=test_config.height,
                width=test_config.width,
                num_frames=test_config.num_frames,
                cfg_scale=test_config.cfg_scale,
                seed=0,
                tiled=test_config.tile,
                verbose=test_config.verbose,
                enable_apg=test_config.verbose,
                parrallel_config=test_config.parallel_configs
            )
           
            if len(result) == 2:
                [output, history_frames] = result
            else:
                output = result[0]
                history_frames = []
            export_to_video(output, str(save_path), fps=test_config.save_fps, quality=6)
            
            frame_intervals = test_config.parallel_configs["frame_intervals"]
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
    args,
    config
):
    inference_case_list = []
    for test_name, config_data in config["inference_cases"].items():
        inference_config = InferenceConfig(
            prompt=config_data["prompt"],
            ref_images=config_data["ref_images"],
            **config["inference_configs"],
        )
        inference_case_list.append((test_name, inference_config))


    import torch.multiprocessing as mp
    from functools import partial
    mp.spawn(
        partial(
            inference_worker,
            args=args,
            inference_case_list=inference_case_list,
            config=config,
        ),
        nprocs=args.world_size,
        join=True,
    )

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default='./benchmarks/inference_configs.json')
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--world_size", type=int, default=8)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    try:
        args = parse_args()
        with open(args.config_path, 'r') as f:
            config = json.load(f)
        run_inference_pipeline(
            args, 
            config,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
