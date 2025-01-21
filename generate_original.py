import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from diffusers import DiffusionPipeline
from experiments.class_load import get_all_categories
from compel import Compel
from experiments.util import set_random_seed, get_unique_filename

# Function to parse command-line arguments
def parse_args():
    """
    Parse the command-line arguments using argparse.
    
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Stable Diffusion Image Generation")

    # Model and device setup
    parser.add_argument('--model_id', type=str, default="./pretrained/stable-diffusion-2-1", help='Pretrained model ID')
    parser.add_argument('--gpu_index', type=int, default=4, help='GPU index (set -1 for CPU)')
    parser.add_argument('--number_loop', type=int, default=30, help='Number of loops for each class')

    # Image generation settings
    parser.add_argument('--num_steps', type=int, default=25, help='Number of diffusion steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale (higher value = stronger adherence to the prompt)')

    # Output directory for images
    parser.add_argument('--output_dir', type=str, default='../generate/output_original_image', help='Directory for saving generated images')
    
    return parser.parse_args()

# Setup device and pipeline based on parsed arguments
args = parse_args()

# Setup device (CPU or GPU)
device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() and args.gpu_index >= 0 else "cpu")

# Load the model and pipeline
pipeline = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float32).to(device)
pipeline.set_progress_bar_config(disable=True, leave=True)

# Setup Compel for text embedding
compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)

# Create the output directory if it doesn't exist
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Retrieve all categories using the function provided in args
all_categories = get_all_categories()

# Loop through all categories and generate images
for classk in tqdm(all_categories, desc="Original image generation"):
    # Generate embedding for the current class prompt
    embedding1 = compel_proc(f"A photo of a {classk}")

    for i in range(args.number_loop):
        # Set a random seed for this iteration
        set_random_seed(i)

        # Generate the image using the diffusion pipeline
        image = pipeline(
            prompt_embeds=embedding1,  # Use the first prompt embedding
            num_inference_steps=args.num_steps,  # Number of diffusion steps
            guidance_scale=args.guidance_scale,  # Guidance scale for adherence to the prompt
        ).images[0]  # Retrieve the first image in the batch

        # Create a unique output file path for the generated image
        category_output_dir = output_dir / classk  # Ensure class-specific folder exists
        category_output_dir.mkdir(parents=True, exist_ok=True)  # Create class-specific directory if it doesn't exist
        
        image_output_path = get_unique_filename(category_output_dir, f'{classk}', f'{i}.png')

        # Save the generated image
        image.save(image_output_path)
        print(f"Image for {classk} saved to {image_output_path}")
