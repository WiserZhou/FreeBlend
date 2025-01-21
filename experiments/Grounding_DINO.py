import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image

def process_image(image_path, text, device, model_id = "./pretrained/grounding-dino-tiny"):
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    image = Image.open(image_path)

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    
    scores = results[0]['scores']
    
    if scores.numel() > 0:  # 确保有元素
        average_score = torch.mean(scores).item()
    else:
        average_score = 0
    
    return average_score