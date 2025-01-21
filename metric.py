import torch
import warnings
import numpy as np
import argparse
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from experiments.class_load import get_mix_categories
from experiments.util import get_image_paths, define_prompt
from experiments.hpsv2.img_score import score
from experiments.util import setup_device
from torchmetrics.multimodal import CLIPImageQualityAssessment
from tqdm import tqdm
from experiments.Grounding_DINO import process_image

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

# Load CLIP and YOLO models
def load_models(gpu_id=0, clip_model_name="./pretrained/clip-vit-base-patch32"):
    """
    Loads both the CLIP model and the YOLO model once.
    """
    clip_model = CLIPModel.from_pretrained(clip_model_name, cache_dir='./pretrained/').cuda(gpu_id)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name, cache_dir='./pretrained/')

    return clip_model, clip_processor

# Get CLIP score for a given image and prompt
def get_clip_score(image_path, prompt, clip_model, clip_processor, gpu_id=0):
    """
    Calculates the CLIP score for a given image and text prompt.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        clip_inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True)
        clip_inputs = {k: v.cuda(gpu_id) for k, v in clip_inputs.items()}
        outputs = clip_model(**clip_inputs)
        return outputs.logits_per_image.abs().cpu().detach().numpy()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return np.array([0])

# Calculate CLIP score differences for mixed images
def calculate_clip_scores(original_image_dir, mixed_image_dir, mix_categories_to_show, clip_model, clip_processor, gpu_id=0):
    """
    Calculate CLIP score differences between mixed and original images.
    """
    mix_object_counts = []

    for mix_category in tqdm(mix_categories_to_show, desc="Calculating CLIP scores"):
        category1, category2 = mix_category.split('_')
        mix_image_paths = get_image_paths(mixed_image_dir, [mix_category])[mix_category]
        original_image_paths1 = get_image_paths(original_image_dir, [category1])[category1]
        original_image_paths2 = get_image_paths(original_image_dir, [category2])[category2]
        
        object_counts = []
        
        for mix_image_path in mix_image_paths:
            score1_mix = get_clip_score(mix_image_path, define_prompt(category1), clip_model, clip_processor, gpu_id)
            score2_mix = get_clip_score(mix_image_path, define_prompt(category2), clip_model, clip_processor, gpu_id)
            score1_original = get_clip_score(original_image_paths1[0], define_prompt(category1), clip_model, clip_processor, gpu_id)
            score2_original = get_clip_score(original_image_paths2[0], define_prompt(category2), clip_model, clip_processor, gpu_id)

            mix_score = score1_mix + score2_mix
            original_score = score1_original + score2_original
            score_difference = abs(mix_score - original_score)
            object_counts.append(score_difference)
            
        mean_count = np.mean(object_counts)
        std_count = np.std(object_counts)

        mix_object_counts.append((mix_category, mean_count, std_count))

        print(f"{mix_category} - Mean CLIP Score Separation Difference: {mean_count}, Std: {std_count}")

    # Calculate overall mean and std for all categories
    overall_mean_count = np.mean([mean for _, mean, _ in mix_object_counts])
    overall_std_count = np.std([mean for _, mean, _ in mix_object_counts])

    print("\nOverall Mean CLIP Score Separation Difference:", overall_mean_count)
    print("Overall Std CLIP Score Separation Difference:", overall_std_count)

    print('--------------------------------')
    print(overall_mean_count, overall_std_count)

    return mix_object_counts, overall_mean_count, overall_std_count

def calculate_hps(mixed_image_dir, mix_categories_to_show, gpu_id=0):

    mix_object_counts = []
    for mix_category in tqdm(mix_categories_to_show, desc="Calculating HPS scores"):    
        mix_image_paths = get_image_paths(mixed_image_dir, [mix_category])[mix_category]
        object_counts = []
        category1, category2 = mix_category.split('_')
        # prompt = f"a photo of {category1} and {category2} mixed together"
        prompt = f"a photo of a blended object combining mixed features from {category1} and {category2}"
        for image_path in mix_image_paths:
            # Run YOLO object detection on the image   
            hps_score = score(image_path, prompt, gpu_id=gpu_id)
            object_counts.append(hps_score)
        # Calculate mean and std deviation for this category
        mean_count = np.mean(object_counts)
        std_count = np.std(object_counts)
        mix_object_counts.append((mix_category, mean_count, std_count))
        print(f"{mix_category} - Mean HPS Count: {mean_count}, Std: {std_count}")
        # 不再逐步记录，而是在后面统一处理

    # Calculate overall mean and std deviation
    overall_mean_count = np.mean([mean for _, mean, _ in mix_object_counts])
    overall_std_count = np.std([mean for _, mean, _ in mix_object_counts])

    print("\nOverall Mean HPS Count:", overall_mean_count)
    print("Overall Std HPS Count:", overall_std_count)

    print('--------------------------------')
    print(overall_mean_count, overall_std_count)

    return mix_object_counts, overall_mean_count, overall_std_count

def calculate_dino(mixed_image_dir, mix_categories_to_show, gpu_id=0):

    device = setup_device(gpu_id)
    mix_object_counts = []
    for mix_category in tqdm(mix_categories_to_show, desc="Calculating DINO scores"):    
        mix_image_paths = get_image_paths(mixed_image_dir, [mix_category])[mix_category]
        object_counts = []
        category1, category2 = mix_category.split('_')
        # prompt = f"a photo of {category1} and {category2} mixed together"
        prompt = f"a {category1} with blending features from {category2}. a {category2} with blending features from {category1}."
        for image_path in mix_image_paths:
            # Run YOLO object detection on the image   
            dino_score = process_image(image_path, prompt, device)
            object_counts.append(dino_score)
        # Calculate mean and std deviation for this category
        mean_count = np.mean(object_counts)
        std_count = np.std(object_counts)
        mix_object_counts.append((mix_category, mean_count, std_count))
        print(f"{mix_category} - Mean DINO Count: {mean_count}, Std: {std_count}")

    # Calculate overall mean and std deviation
    overall_mean_count = np.mean([mean for _, mean, _ in mix_object_counts])
    overall_std_count = np.std([mean for _, mean, _ in mix_object_counts])

    print("\nOverall Mean DINO Count:", overall_mean_count)
    print("Overall Std DINO Count:", overall_std_count)

    print('--------------------------------')
    print(overall_mean_count, overall_std_count)

    return mix_object_counts, overall_mean_count, overall_std_count

# https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_iqa.html
def return_CLIP_IQA(prompts):
    clip_iqa = CLIPImageQualityAssessment(model_name_or_path = "./pretrained/clip-vit-base-patch32", 
                                        prompts=prompts)
    return clip_iqa

def calculate_CLIP_IQA(mixed_image_dir, mix_categories_to_show):
    prompts = (("mixed", "dull"), ("blending features from two different objects", "natural object from one object"))
    clip_iqa = return_CLIP_IQA(prompts)
    
    mix_object_counts = []
    all_mixed_scores = []
    all_blending_scores = []
    
    for mix_category in tqdm(mix_categories_to_show, desc="Calculating CLIP IQA scores"):    
        mix_image_paths = get_image_paths(mixed_image_dir, [mix_category])[mix_category]
        mixed_scores = []
        blending_scores = []
        
        for image_path in mix_image_paths:
            try:
                image = torch.from_numpy(np.array(Image.open(image_path))).permute(2, 0, 1).unsqueeze(0).float()
                scores = clip_iqa(image)
                mixed_scores.append(scores['user_defined_0'].item())
                blending_scores.append(scores['user_defined_1'].item())
                
                all_mixed_scores.append(scores['user_defined_0'].item())
                all_blending_scores.append(scores['user_defined_1'].item())
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue
        
        mean_mixed = np.mean(mixed_scores)
        std_mixed = np.std(mixed_scores)
        mean_blending = np.mean(blending_scores)
        std_blending = np.std(blending_scores)
        
        # 计算总体平均分
        mean_count = (mean_mixed + mean_blending) / 2
        std_count = (std_mixed + std_blending) / 2
        
        mix_object_counts.append((mix_category, mean_count, std_count))
        print(f"{mix_category} - Mean CLIP-IQA Scores:")
        print(f"  Mixed: {mean_mixed:.3f} (±{std_mixed:.3f})")
        print(f"  Blending: {mean_blending:.3f} (±{std_blending:.3f})")
        print(f"  Overall: {mean_count:.3f} (±{std_count:.3f})")

    # Calculate overall mean and std deviation
    overall_mean_count = np.mean([mean for _, mean, _ in mix_object_counts])
    overall_std_count = np.std([mean for _, mean, _ in mix_object_counts])

    overall_mean_mixed = np.mean(all_mixed_scores)
    overall_std_mixed = np.std(all_mixed_scores)
    overall_mean_blending = np.mean(all_blending_scores)
    overall_std_blending = np.std(all_blending_scores)

    print("\nOverall Mean CLIP-IQA Score:", overall_mean_count)
    print("Overall Std CLIP-IQA Score:", overall_std_count)
    print("Overall Mean Mixed Score:", overall_mean_mixed)
    print("Overall Std Mixed Score:", overall_std_mixed)
    print("Overall Mean Blending Score:", overall_mean_blending)
    print("Overall Std Blending Score:", overall_std_blending)

    print('--------------------------------')
    print(overall_mean_count, overall_std_count, 
        overall_mean_mixed, overall_std_mixed, 
        overall_mean_blending, overall_std_blending)

    return mix_object_counts, overall_mean_count, overall_std_count

# Main function to compute all metrics
def compute_metrics(original_image_dir, mixed_image_dir, mix_categories_to_show=None, 
                    output_csv_filename="metrics_results.csv", gpu_id=0, 
                    metrics_to_compute=None):
    """
    Compute selected metrics for mixed categories.
    
    Args:
        original_image_dir: Directory with original images
        mixed_image_dir: Directory with mixed images
        mix_categories_to_show: Categories to analyze
        output_csv_filename: Output CSV filename
        gpu_id: GPU ID for computation
        metrics_to_compute: List of metrics to compute. Options: 
        ['clip', 'hps', 'clip_iqa', 'dino'] If None, computes all metrics.
    """
    # torch.set_num_threads(20)
    
    if mix_categories_to_show is None:
        mix_categories_to_show = get_mix_categories()
    
    if metrics_to_compute is None:
        metrics_to_compute = ['clip', 'hps', 'clip_iqa', 'dino']
    
    update_mix = []
    for mix_category in mix_categories_to_show:
        if isinstance(mix_category, tuple):
            mix_category = '_'.join(mix_category)
        update_mix.append(mix_category)
    
    mix_categories_to_show = update_mix

    # Only load CLIP model if needed
    clip_model, clip_processor = None, None
    if 'clip' in metrics_to_compute:
        clip_model, clip_processor = load_models(gpu_id)
        calculate_clip_scores(
            original_image_dir, mixed_image_dir, mix_categories_to_show, 
            clip_model, clip_processor, gpu_id
        )
    
    if 'hps' in metrics_to_compute:
        calculate_hps(mixed_image_dir, mix_categories_to_show, gpu_id)
    
    if 'clip_iqa' in metrics_to_compute:
        calculate_CLIP_IQA(mixed_image_dir, mix_categories_to_show)
    
    if 'dino' in metrics_to_compute:
        calculate_dino(mixed_image_dir, mix_categories_to_show, gpu_id)

# Command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Compute metrics for mixed categories images")
    parser.add_argument("--original_image_dir", required=True, help="Directory with original images")
    parser.add_argument("--mixed_image_dir", required=True, help="Directory with mixed images")
    parser.add_argument("--output_csv_filename", default="metrics_results.csv", help="Output CSV filename")
    parser.add_argument("--mix_categories_to_show", nargs='*', default=None, help="Categories to analyze (e.g., 'cat_dog')")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID for FID calculation")
    parser.add_argument("--metrics_to_compute", nargs='*', default=None, help="Metrics to compute (e.g., 'clip', 'hps', 'aes', 'clip_iqa')")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    compute_metrics(
        original_image_dir=args.original_image_dir,
        mixed_image_dir=args.mixed_image_dir,
        mix_categories_to_show=args.mix_categories_to_show,
        output_csv_filename=args.output_csv_filename,
        gpu_id=args.gpu_id,
        metrics_to_compute=args.metrics_to_compute
    )

