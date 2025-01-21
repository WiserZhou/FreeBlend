import argparse
from pathlib import Path
from PIL import Image
import torch
from compel import Compel
from tqdm import tqdm
from typing import List, Dict
from experiments.util import get_unique_filename, setup_device, set_random_seed
from experiments.class_load import get_mix_categories
from models.CustomSD import CustomSDPipeline
from models.CustomUNet2DCM import CustomUNet2DConditionModel
from models.StageUnclip import StageStableUnCLIPImg2ImgPipeline
import sys
from metric import compute_metrics


IMG_PIPELINE_TYPES = (StageStableUnCLIPImg2ImgPipeline,)

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the image generation experiment script.
    """
    parser = argparse.ArgumentParser(description="Image generation experiment script")

    # Model and device settings
    parser.add_argument(
        "--model_type", type=str, default="stable_diffusion",
        help="Choose model type: 'stable_diffusion', 'unclip', or 'custom_unclip'"
    )
    parser.add_argument("--model_id", type=str, default="./pretrained/stable-diffusion-2-1", 
                        help="Pre-trained model ID (used with 'stable_diffusion')")
    parser.add_argument("--unclip_model_id", type=str, default="./pretrained/stable-diffusion-2-1-unclip", 
                        help="Pre-trained UnCLIP model ID (used with 'unclip')")
    parser.add_argument("--gpu_index", type=int, default=6, help="GPU index (set to -1 to use CPU)")
    parser.add_argument("--number_loop", type=int, default=30, help="Number of images to generate per category")

    # Image generation settings
    parser.add_argument("--num_steps", type=int, default=25, help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, 
                        help="Guidance scale (higher values mean stronger adherence to the prompt)")

    # Output directory
    parser.add_argument("--output_dir", type=str, default="../generate/output_blend/blend", 
                        help="Directory to save generated images")

    # Category JSON file path
    parser.add_argument("--categories_file", type=str, default="experiments/categories.json", 
                        help="Path to JSON file containing category list")
    
    # Original images directory
    parser.add_argument("--original_images_dir", type=str, default="../generate/output_original_image", 
                        help="Base directory containing original category images")

    # Text embedding method
    parser.add_argument('--text_embeding', type=str, default='None', help='Text embedding method: "compel" or "unet or None"')
    parser.add_argument('--avg_img', type=int, default=1, help='Average the two images or use unet for two period')
    parser.add_argument('--interpolation_type', type=str, default='decline', help='Interpolation type: "decline", "invariant", or "increase"')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma for the blending latent of img_1 and img_2")
    
    return parser.parse_args()

def load_pipeline_and_model(args, device):
    """
    Load the appropriate model pipeline based on the selected model type.
    Also loads the Compel text processor and custom UNet model if necessary.
    """
    pipeline, compel_proc = None, None

    # Load the model pipeline based on the selected model type
    if args.model_type == "stable_diffusion":
        pipeline = CustomSDPipeline.from_pretrained(args.model_id, torch_dtype=torch.float32).to(device)
    elif args.model_type == "stage_unclip":
        pipeline = StageStableUnCLIPImg2ImgPipeline.from_pretrained(args.unclip_model_id,
                                                        torch_dtype=torch.float32, safety_checker=None).to(device)
    else:
        raise ValueError("Unsupported model type. Choose 'stable_diffusion', 'unclip', or 'custom_unclip'.")

    # Initialize the text embedding processor (Compel)
    compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
    pipeline.set_progress_bar_config(disable=True, leave=True)
    
    # Load and replace the UNet model for custom pipelines
    # if isinstance(pipeline, (CustomSDPipeline, CustomStableUnCLIPImg2ImgPipeline)):
    custom_unet = CustomUNet2DConditionModel(**pipeline.unet.config).to(device)
    custom_unet.load_state_dict(pipeline.unet.state_dict())
    pipeline.unet = custom_unet.eval()  # Set to evaluation mode

    return pipeline, compel_proc


def load_class_images(base_dir: str) -> Dict[str, List[Path]]:
    """
    Load image paths for each class from the specified base directory.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    class_images = {}
    for class_dir in base_path.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.[jpJP]*"))  # Match common image formats
            class_images[class_dir.name] = images

    return class_images


def generate_mixed_image(pipeline, compel_proc, args, class1: str, class2: str, original_img1: Image.Image, 
                        original_img2: Image.Image):
    """
    Generate a mixed image based on embeddings from two classes and their original images.
    """
    prompt1 = f"A photo of a {class1}"
    prompt2 = f"A photo of a {class2}"
    embedding1, embedding2 = compel_proc(prompt1), compel_proc(prompt2)

    # Generate mixed image based on the chosen embedding method  
    if args.text_embeding == 'None':
        if isinstance(pipeline, IMG_PIPELINE_TYPES):
            return pipeline(image=original_img1, image_2=original_img2,
                            num_inference_steps=args.num_steps, guidance_scale=args.guidance_scale,
                            avg_img=args.avg_img, interpolation_type=args.interpolation_type, gamma=[args.gamma, 2 - args.gamma]).images[0]
        else:
            raise ValueError("Unsupported text embedding method. Choose 'compel', 'unet', or 'None'.")
    else:
        raise ValueError("Unsupported text embedding method. Choose 'compel', 'unet', or 'None'.")

def combine_images_and_save(pipeline, compel_proc, args, class1: str, class2: str, output_dir: Path, 
                            original_img1, original_img2, next_number_loop: int):
    """
    Generate and save mixed images for the given classes and their original images.
    """

    # Generate multiple mixed images (as specified by 'number_loop')
    for i in range(next_number_loop):
        
        # Generate the mixed image based on the original images and class embeddings
        mixed_image = generate_mixed_image(pipeline, compel_proc, args, class1, class2, 
                                        Image.open(original_img1[i]), Image.open(original_img2[i]))

        # Save the mixed image to the output directory
        mixed_image_filename = f"mixed_{class1}_{class2}_{i}"
        mixed_image_output_path = get_unique_filename(output_dir, mixed_image_filename)
        mixed_image.save(mixed_image_output_path)

def main():
    # Parse command-line arguments
    args = parse_args()
    # set_random_seed(args.seed)
    avg_or_unet = 'avg' if args.avg_img == 1 else 'unet'

    # Set up device (GPU/CPU)
    device = setup_device(args.gpu_index)

    # Load model pipeline and text processor
    pipeline, compel_proc = load_pipeline_and_model(args, device)
    
    # Load images from original categories
    class_images = load_class_images(args.original_images_dir)

    # Load the list of category pairs to mix (from the JSON file)
    mix_categories = get_mix_categories(args.categories_file)
    
    print(f"Mix categories to process: {mix_categories}")

    # Create output directory if it doesn't exist
    args.output_dir = args.output_dir + '_' + args.text_embeding + '_' + args.model_type + '_' + avg_or_unet + '_' + args.interpolation_type
    print(f"Output directory: {args.output_dir}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.text_embeding == 'None' and args.model_type == 'stable_diffusion':
        sys.exit("Terminating process as text embedding is 'None' and model type is 'stable_diffusion'")

    # Generate and save mixed images for each pair of categories
    for class1, class2 in tqdm(mix_categories, desc="Generating mixed images"):
        if class1 in class_images and class2 in class_images:

            # Create a subdirectory for the specific class1_class2 combination
            class_output_dir = output_dir / f"{class1}_{class2}"
            class_output_dir.mkdir(parents=True, exist_ok=True)

            # Check the number of existing images in the subdirectory
            existing_images = list(class_output_dir.glob("*.png"))
            if len(existing_images) >= args.number_loop:
                print(f"Skipping generation for '{class1}_{class2}' as it already has {len(existing_images)} images")
                continue
            else:
                print(f"Generating mixed images for categories: '{class1}' and '{class2}'")
                next_number_loop = args.number_loop - len(existing_images)
                # Combine images and save to the specific subdirectory
                combine_images_and_save(pipeline, compel_proc, args, class1, class2, class_output_dir, 
                                        class_images[class1], class_images[class2], next_number_loop)
        else:
            print(f"Images for class '{class1}' or '{class2}' not found")
    
    # Release GPU memory by clearing pipeline and processor
    del pipeline
    del compel_proc
    torch.cuda.empty_cache()
    
    compute_metrics(
        original_image_dir=args.original_images_dir,
        mixed_image_dir=args.output_dir,
        mix_categories_to_show=mix_categories,
        gpu_id=args.gpu_index,
    )

if __name__ == "__main__":
    main()
