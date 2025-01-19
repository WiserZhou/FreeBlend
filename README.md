# Environment Setup

To set up the environment for this project, follow these steps:

1. **Create a new conda environment** with Python 3.10.15:
    ```bash
    conda create --name Z python=3.10.15
    ```
2. **Activate the environment**:
    ```bash
    conda activate Z
    ```
3. **Install required packages** using pip:
    ```bash
    pip install diffusers==0.31.0
    pip install torch torchvision transformers compel accelerate gpustat matplotlib open-clip-torch clint pycuda einops spacy scipy scikit-learn addict supervision yapf pycocotools jupyter ipywidgets
    ```
4. **Download required models** using the provided scripts:
    - **For model generation**:
        ```bash
        ./download.sh stabilityai/stable-diffusion-2-1
        ./download.sh stabilityai/stable-diffusion-2-1-unclip
        ```
    - **For Attend&Excite**:
        ```bash
        ./download.sh CompVis/stable-diffusion-v1-4
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
5. **Create generate directory in parent folder**:
    ```bash
    cd ..
    mkdir -p generate
    ```
6. **Create subdirectories**:
    ```bash
    mkdir -p generate/output_blend
    mkdir -p generate/output_original_image
    cd blend_concept
    ```

7. **Generate original images**:
    ```bash
    nohup python generate_original.py \
        --gpu_index 5 \
        --num_steps 25 \
        --guidance_scale 7.5 \
        --output_dir "../generate/output_original_image" > out_original.log 2>&1 &
    ```

8. **Generate blend images**:
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
        --text_embeding None --gpu_index 1 \
        --avg_img 0 \
        --output_dir "../generate/output_blend/blend" \
        --interpolation_type decline > out_blend_decline.log 2>&1 &
    ```

# blend_concept_2
# blend_concept_2
