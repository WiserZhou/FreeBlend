# blend_compel_stable_diffusion_avg
nohup python metric.py \
    --original_image_dir /home/hwang/Projects/generate/output_original_image \
    --mixed_image_dir    /home/hwang/Projects/generate/output_blend/blend_compel_stable_diffusion_avg_decline \
    --metrics_to_compute hps \
    --gpu_id 0 > out2metric.log 2>&1 &

# blend_unet_stable_diffusion_avg
nohup python metric.py \
    --original_image_dir /home/hwang/Projects/generate/output_original_image \
    --mixed_image_dir    /home/hwang/Projects/generate/output_blend/blend_unet_stable_diffusion_avg_decline \
    --metrics_to_compute hps \
    --gpu_id 1 > out3metric.log 2>&1 &

# blend_None_stage_unclip_avg
nohup python metric.py \
    --original_image_dir /home/hwang/Projects/generate/output_original_image \
    --mixed_image_dir    /home/hwang/Projects/generate/output_blend/blend_None_stage_unclip_avg_decline \
    --gpu_id 1 > out18metric.log 2>&1 &

# blend_None_stage_unclip_unet
nohup python metric.py \
    --original_image_dir /home/hwang/Projects/generate/output_original_image \
    --mixed_image_dir    /home/hwang/Projects/generate/output_blend/blend_None_stage_unclip_unet_decline \
    --gpu_id 2 > out19metric.log 2>&1 &

# blend_compel_stage_unclip_avg
nohup python metric.py \
    --original_image_dir /home/hwang/Projects/generate/output_original_image \
    --mixed_image_dir    /home/hwang/Projects/generate/output_blend/blend_compel_stage_unclip_avg_decline \
    --gpu_id 0 > out20metric.log 2>&1 &

# blend_compel_stage_unclip_unet
nohup python metric.py \
    --original_image_dir /home/hwang/Projects/generate/output_original_image \
    --mixed_image_dir    /home/hwang/Projects/generate/output_blend/blend_compel_stage_unclip_unet_decline \
    --gpu_id 1 > out21metric.log 2>&1 &

# blend_unet_stage_unclip_avg
nohup python metric.py \
    --original_image_dir /home/hwang/Projects/generate/output_original_image \
    --mixed_image_dir    /home/hwang/Projects/generate/output_blend/blend_unet_stage_unclip_avg_decline \
    --gpu_id 2 > out22metric.log 2>&1 &

# blend_unet_stage_unclip_unet
nohup python metric.py \
    --original_image_dir /home/hwang/Projects/generate/output_original_image \
    --mixed_image_dir    /home/hwang/Projects/generate/output_blend/blend_unet_stage_unclip_unet_decline \
    --gpu_id 0 > out23metric.log 2>&1 &