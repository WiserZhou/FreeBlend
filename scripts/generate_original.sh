nohup python generate_original.py \
    --gpu_index 5 \
    --num_steps 25 \
    --guidance_scale 7.5 \
    --output_dir "../generate/output_original_image"
    > out.log 2>&1 &