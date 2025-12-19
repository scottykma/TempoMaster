# TempoMaster: Efficient Long Video Generation via Next-Frame-Rate Prediction
[![arXiv](https://img.shields.io/badge/arXiv-2511.12578-b31b1b.svg)](https://arxiv.org/abs/2511.12578)
[![Project Page](https://img.shields.io/badge/Project_Page-green)](https://scottykma.github.io/tempomaster-gitpage/)
[![Static Badge](https://img.shields.io/badge/Model-Huggingface-yellow)](https://huggingface.co/Scottttttyy/TempoMaster)

> We present TempoMaster, a novel framework that formulates long video generation as next-frame-rate prediction. Specifically, we first generate a low-frame-rate clip that serves as a coarse blueprint of the entire video sequence, and then progressively increase the frame rate to refine visual details and motion continuity. During generation, TempoMaster employs bidirectional attention within each frame-rate level while performing autoregression across frame rates, thus achieving long-range temporal coherence while enabling efficient and parallel synthesis. Extensive experiments demonstrate that TempoMaster establishes a new state-of-the-art in long video generation, excelling in both visual and temporal quality.

## Overview

![Fig.1](https://scottykma.github.io/tempomaster-gitpage/static/images/teaser.jpg)

TempoMaster is a video diffusion model capable of generating videos at various frame rates. 
The model first generates a low-frame-rate video as a global blueprint. It then uses the existing frames as temporal anchors to infer and insert additional frames in between, progressively upsampling the video to higher frame rates.
This approach effectively structures long-term temporal dynamics and mitigates the issue of visual drifting caused by error accumulation.

## Demo Videos

<table>
  <tr>
    <td align="center" width="832">
      <video 
        src="https://scottykma.github.io/tempomaster-gitpage/static/videos/horse_seed_42_cfg5&3_shift_8&6_step_50&32_0.mp4" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>The generated lowest frame rate (6fps) video</em>
    </td>
  </tr>
</table>
<table>
  <tr>
    <td align="center" width="832">
      <video 
        src="https://scottykma.github.io/tempomaster-gitpage/static/videos/horse_seed_42_cfg5&3_shift_8&6_step_50&32_1.mp4" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>And its corresponding 24fps video</em>
    </td>
  </tr>
</table>

## Quick Start

1. Clone this repo
```bash
git clone https://github.com/scottykma/TempoMaster.git
cd TempoMaster
```
2. Install the requirements
```bash
pip install -r requirements.txt
```
3. Download the checkpoints from huggingface and place them as below:
```bash
TempoMaster
│       
├─checkpoints
│  │  wan_video_dit_bf16.bin
│  │
│  └─encoders
│      │  models_t5_umt5-xxl-enc-bf16.pth
│      │  Wan2.1_VAE.pth
│      │
│      └─google
│          └─umt5-xxl
│                  special_tokens_map.json
│                  spiece.model
│                  tokenizer.json
│                  tokenizer_config.json   
│
│ ...
```
4. Run the following command
```bash
python infer.py --world-size 8
```

## More Details

The inference config is listed in `inference_configs.json`.

```json
"inference_configs": {
        "height": 480, 
        "width": 832,
        "num_frames": 109, # the number of frames the model need to generate at once
        "cfg_scale": 8, # it could be a large scale for APG, if use_apg == false, it shoud be a lower number for cfg
        "tile": false,
        "save_fps": 24,
        "negative_prompt": "jittery, shaky, fliker, unstable, static, ultra slow motion, distorted, blurry details, many people in the background worst quality, low quality, ugly, poorly drawn legs, poorly drawn face, three arms, deformed, mutilated, disfigured, malformed limbs, fused fingers, three legs",
        "use_apg": true, # whether to use apg or not (cfg)
        "verbose": false, # whether to return the lower-frame-rate videos
        "parallel_configs": {
            "frame_intervals": [4, 2, 1], # generate videos at 6, 12, 24fps, can be changed to [4, 1] or [2, 1]
            "steps_per_stage": [50, 32, 32], # fewer step for the following stage with more conditions (anchor frames)
            "shift_per_stage": [5, 3, 3] # lower shift value for the following stage with more conditions
        }
    }, 
```

## Inference with Multiple GPUs

We will release inference scripts with context parallel in the future. However, it is also easy to use data parallel to enable parallel generation.

In the function `prepare_condition_video`, we divide the whole frame sequence into chunks with the same length. These chunks are stacked into a batch and fed to the model in `denoise_chunk`:
```python
noise_pred_posi = self.dit(
                    latents, 
                    timestep=timestep, 
                    freqs=freqs_batch,
                    context=context.expand(b, -1, -1),
                    y=image_emb["y"]
                )
```
So it is easy to use dp as below:
```python
model = nn.DataParallel(self.dit, device_ids=[... Your GPU IDs])
...

noise_pred_posi = model(
    ...
)
noise_pred_nage = ...
```

## Cite


    @article{tempomaster2025,
        title={TempoMaster: Efficient Long Video Generation via Next-Frame-Rate Prediction}, 
        author={Yukuo Ma and Cong Liu and Junke Wang and Junqi Liu and Haibin Huang and Zuxuan Wu and Chi Zhang and Xuelong Li},
        year={2025},
        eprint={2511.12578},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2511.12578}, 
    }
