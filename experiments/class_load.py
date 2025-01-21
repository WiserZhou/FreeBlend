import os
from glob import glob
import json
import itertools

# 从文件读取 JSON 数据
def load_json_data(file_path):
    """
    Load JSON data from a file.
    
    :param file_path: The path to the JSON file.
    :return: The parsed JSON data as a dictionary.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Failed to decode the JSON data in {file_path}.")
        return None

# 获取所有类（类别）
def get_all_categories(file_path = None):
    """
    Read the JSON file and return a list of all categories from 'all_categories_list'.
    
    :param file_path: The path to the JSON file.
    :return: A list of all categories.
    """
    if file_path is None:
        file_path = 'experiments/categories.json'
        
    data = load_json_data(file_path)
    if data:
        return data.get('all_categories_list', [])
    else:
        return []

# 获取所有混合类（C20, 2 的组合）
def get_mix_categories(file_path=None):
    """
    Read the JSON file and return a list of mixed categories (pairs of categories).
    
    :param file_path: The path to the JSON file.
    :return: A list of mixed categories in 'category1_category2' format.
    """
    if file_path is None:
        file_path = 'experiments/categories.json'
        
    data = load_json_data(file_path)
    if data:
        all_categories = data.get('all_categories_list', [])
        
        # Generate all combinations of 2 categories (C20, 2)
        combinations = list(itertools.combinations(all_categories, 2))
        
        # Convert the combinations to the "category1_category2" format
        mix_categories = [(pair[0], pair[1]) for pair in combinations]
        
        return mix_categories
    else:
        return []

def load_images_from_categories(base_folder, mix_folder=None, kind_list=None, n_images=30):
    """
    Load images from the specified base folder and mix folder.
    If a kind list is provided, only load images from the specified kinds.
    If a mix folder is provided, load images from the mix folder as well.
    The number of images to load per category is specified by n_images.
    
    :param base_folder: The base folder to load images from.
    :param mix_folder: The mix folder to load images from. If None, do not load images from a mix folder.
    :param kind_list: A list of kinds to load images from. If None, load images from all kinds.
    :param n_images: The number of images to load per category.
    :return: A tuple of two lists - the first list contains all image paths, the second list contains the corresponding categories.
    """
    all_image_paths = []
    categories = []
    
    # Load images from the base folder
    category_folders = [f for f in os.listdir(base_folder) 
                        if os.path.isdir(os.path.join(base_folder, f))
                        and (kind_list is None or f in kind_list)]
    
    for category in category_folders:
        category_path = os.path.join(base_folder, category)
        image_files = glob(os.path.join(category_path, "*.jpg")) + \
                        glob(os.path.join(category_path, "*.png"))
        category_images = image_files[:n_images]
        all_image_paths.extend(category_images)
        categories.extend([category] * len(category_images))
    
    # Load images from the mix folder
    if mix_folder and os.path.exists(mix_folder):
        mix_folders = [f for f in os.listdir(mix_folder) 
                        if os.path.isdir(os.path.join(mix_folder, f))]
        
        for mix_category in mix_folders:
            mix_path = os.path.join(mix_folder, mix_category)
            mix_images = glob(os.path.join(mix_path, "*.jpg")) + \
                        glob(os.path.join(mix_path, "*.png"))
            mix_images = mix_images[:n_images]
            all_image_paths.extend(mix_images)
            categories.extend([mix_category] * len(mix_images))
    
    return all_image_paths, categories