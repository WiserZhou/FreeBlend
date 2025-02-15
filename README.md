# FreeBlend: Advancing Concept Blending with Staged Feedback-Driven Interpolation Diffusion

<div align="center">

[![a](https://img.shields.io/badge/Website-FreeBlend-blue)](https://petershen-csworld.github.io/FreeBlend/)
[![arXiv](https://img.shields.io/badge/arXiv-2502.05606-red)](https://arxiv.org/abs/2502.05606)
</div>

> #### [**FreeBlend**: Advancing Concept Blending with Staged Feedback-Driven Interpolation Diffusion](https://arxiv.org/abs/2502.05606)
> ##### [Yufan Zhou*](https://wiserzhou.github.io/), [Haoyu Shen*](https://github.com/), [Huan Wang](https://huanwang.tech/) ("*" denotes equal contribution)



## Environment Setup

To set up the environment for this project, follow these steps:

1. **Create a new conda environment** with Python 3.10.15:
    ```bash
    conda create --name FreeBlend python=3.10.15
    ```
2. **Activate the environment**:
    ```bash
    conda activate FreeBlend
    ```
3. **Install required packages** using pip:
    ```bash
    pip install diffusers==0.31.0
    pip install torch torchvision transformers compel accelerate gpustat matplotlib open-clip-torch clint pycuda einops spacy scipy scikit-learn addict supervision yapf pycocotools jupyter ipywidgets torchmetrics
    ```
4. **Download required models** using the provided scripts:
    - **For model generation**:
        ```bash
        ./download.sh stabilityai/stable-diffusion-2-1
        ./download.sh stabilityai/stable-diffusion-2-1-unclip
        ```
    - **For HPS**:
        ```bash
        ./download.sh laion/CLIP-ViT-H-14-laion2B-s32B-b79K
        ```
    - **For CLIP-IQA**:
        ```bash
        ./download.sh openai/clip-vit-base-patch32
        ```
    - **For DINO**:
        ```bash
        ./download.sh IDEA-Research/grounding-dino-tiny
        ```
5. **Quick start with jupyter notebook**:

    Navigate to and run `stage_unclip.ipynb` to test the functionality.

6. **Create generate directory in parent folder**:
    ```bash
    cd ..
    mkdir -p generate
    ```
7. **Create subdirectories**:
    ```bash
    mkdir -p generate/output_blend
    mkdir -p generate/output_original_image
    cd blend_concept
    ```

8. **Generate original images**:
    ```bash
    # Generate original images with specified parameters
    nohup python generate_original.py \
        --gpu_index 0 \
        --num_steps 25 \
        --guidance_scale 7.5 \
        --output_dir "../generate/output_original_image" > out_original.log 2>&1 &
    ```

    **Note**: If you encounter the error `undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12`:
    - Option 1: Reinstall torch and torchvision packages
    - Option 2: Set LD_LIBRARY_PATH:
    ```bash
    export LD_LIBRARY_PATH=/home/user/miniconda3/envs/Z/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
    ```
    For more details, see: https://github.com/pytorch/pytorch/issues/131312

9. **Generate blend images and compute metrics**:
    ```bash
    number_loop=30
    num_steps=25
    guidance_scale=7.5
    categories_file="experiments/categories.json"
    original_images_dir="../generate/output_original_image"

    common_params="--number_loop $number_loop --num_steps $num_steps \
                --guidance_scale $guidance_scale \
                --categories_file $categories_file \
                --original_images_dir $original_images_dir"

    nohup python generate_blend.py $common_params --model_type stage_unclip \
        --text_embeding None --gpu_index 0 \
        --avg_img 0 \
        --output_dir "../generate/output_blend/blend" \
        --interpolation_type decline > out_blend_decline.log 2>&1 &
    ```

10. **Compute metrics**:
    ```bash
    nohup python metric.py \
    --original_image_dir ../generate/output_original_image \
    --mixed_image_dir    ../generate/output_blend/blend_None_stage_unclip_unet_decline \
    --gpu_id 0 > out_metric.log 2>&1 &
    ```

## Citation

```
@article{zhou2025freeblend,
  title={FreeBlend: Advancing Concept Blending with Staged Feedback-Driven Interpolation Diffusion},
  author={Zhou, Yufan and Shen, Haoyu and Wang, Huan},
  journal={arXiv preprint arXiv:2502.05606},
  year={2025}
}
```

## Acknowledgement

We appreciate the authors of [HPSv2](https://github.com/tgxs002/HPSv2), [CLIP-IQA](https://github.com/IceClear/CLIP-IQA), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip), and [MagicMix](https://github.com/daspartho/MagicMix) to share their code.
