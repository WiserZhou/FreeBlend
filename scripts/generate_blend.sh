#!/bin/bash

# Parameters
number_loop=30
num_steps=25
guidance_scale=7.5
categories_file="experiments/categories.json"
original_images_dir="/home/hwang/Projects/generate/output_original_image"

common_params="--number_loop $number_loop --num_steps $num_steps \
            --guidance_scale $guidance_scale \
            --categories_file $categories_file \
            --original_images_dir $original_images_dir"

# Example command

# stable_diffusion

nohup python generate_blend.py $common_params --model_type stable_diffusion --text_embeding None --gpu_index 0 > out1.log 2>&1 &

nohup python generate_blend.py $common_params --model_type stable_diffusion --text_embeding compel --gpu_index 1 > out2.log 2>&1 &

nohup python generate_blend.py $common_params --model_type stable_diffusion --text_embeding unet --gpu_index 3 > out6.log 2>&1 &

nohup python generate_blend.py $common_params --model_type stable_diffusion --text_embeding alternate --gpu_index 3 > out4.log 2>&1 &

# stage_unclip

nohup python generate_blend.py $common_params --model_type stage_unclip --text_embeding None --gpu_index 0 --avg_img 0 > out2.log 2>&1 &

nohup python generate_blend.py $common_params --model_type stage_unclip --text_embeding None --gpu_index 0 > out3.log 2>&1 &

nohup python generate_blend.py $common_params --model_type stage_unclip --text_embeding compel --gpu_index 2 --avg_img 0 > out5.log 2>&1 &

nohup python generate_blend.py $common_params --model_type stage_unclip --text_embeding compel --gpu_index 3 > out6.log 2>&1 &

nohup python generate_blend.py $common_params --model_type stage_unclip --text_embeding unet --gpu_index 0 > out7.log 2>&1 &

nohup python generate_blend.py $common_params --model_type stage_unclip --text_embeding unet --gpu_index 1 --avg_img 0 > out8.log 2>&1 &