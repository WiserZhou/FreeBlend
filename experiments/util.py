import torch
import random
import numpy as np
import time
import os
from models.CustomUNet2DCM import CustomUNet2DConditionModel

# Set random seed function to ensure different seeds for each generation loop
def set_random_seed(seed=None):
    """
    Set random seed for reproducibility across different libraries.
    
    :param seed: An optional seed value. If None, the seed is based on the current time.
    :return: The actual seed used.
    """
    if seed is None:
        # Use the current timestamp to generate a seed
        seed = int(time.time()) % (2**32)  # Limiting seed range to 32-bit unsigned integer
    torch.manual_seed(seed)  # Set PyTorch random seed
    torch.cuda.manual_seed_all(seed)  # Set all GPU random seeds
    random.seed(seed)  # Set Python random seed
    np.random.seed(seed)  # Set NumPy random seed
    return seed

def get_unique_filename(base_path, base_name, extension='.png'):
    """
    Ensures the output file is unique by appending a number if a file already exists.
    
    :param base_path: The directory where the file will be saved.
    :param base_name: The base name of the file (without extension).
    :param extension: The file extension (e.g., '.png').
    :return: A unique file path.
    """
    counter = 0
    while True:
        suffix = f"_{counter}" if counter > 0 else ""
        full_path = base_path / f"{base_name}{suffix}{extension}"
        if not full_path.exists():
            return full_path
        counter += 1

# -------------------- 设备和管道设置 --------------------

def setup_device(gpu_index: int) -> torch.device:
    """
    设置计算设备。

    Args:
        gpu_index (int): 要使用的GPU索引。设置为-1使用CPU。

    Returns:
        torch.device: 计算设备。
    """
    if gpu_index >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_index}")
        # print(f"使用GPU: {gpu_index}")
    else:
        device = torch.device("cpu")
        # print("使用CPU")
    return device

# Get image paths for a given category
def get_image_paths(image_dir, categories):
    """
    Retrieves image paths for the given categories from the specified directory.
    """
    image_paths = {}
    for category in categories:
        category_dir = os.path.join(image_dir, category)
        if not os.path.exists(category_dir):
            print(f"Directory {category_dir} not found.")
            image_paths[category] = []
        else:
            image_paths[category] = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
    return image_paths

def define_prompt(category, category2=None):
    if category2 is None:
        return f"a photo of a {category}"
    else:
        return f"a photo of a blended object combining {category} and {category2}"

def change_UNet(pipeline, gpu_id):
    
    device = setup_device(gpu_id)
    
    # 优化UNet模型
    custom_unet = CustomUNet2DConditionModel(**pipeline.unet.config).to(device)
    custom_unet.load_state_dict(pipeline.unet.state_dict())
    custom_unet = custom_unet.to(dtype=torch.float32) 
    pipeline.unet = custom_unet.eval()  # 设置为评估模式