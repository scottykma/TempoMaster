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
|   
│ ...
```
4. 
