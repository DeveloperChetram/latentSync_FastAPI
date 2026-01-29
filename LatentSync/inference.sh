# #!/bin/bash
# python -m scripts.inference \
#     --unet_config_path "configs/unet/stage2.yaml" \
#     --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
#     --inference_steps 20 \
#     --guidance_scale 3 \
#     --enable_deepcache \
#     --video_path "assets/demo1_video.mp4" \
#     --audio_path "assets/demo1_audio.wav" \
#     --video_out_path "video_out.mp4"

#!/bin/bash

# Optional memory optimization
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# python -m scripts.inference \
#     --unet_config_path "configs/unet/stage2_512.yaml" \
#     --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
#     --inference_steps 20 \
#     --guidance_scale 1.7 \
#     --enable_deepcache \
#     --video_path "assets/test_25fps.mp4" \
#     --audio_path "assets/test.mp3" \
#     --video_out_path "video_out_highres_new.mp4"

#!/bin/bash

# Memory optimization (Keep this for L4)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m scripts.inference \
    --unet_config_path "configs/unet/stage2_512.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --inference_steps 10 \
    --guidance_scale 1.6 \
    --video_path "assets/girl2.mp4" \
    --audio_path "assets/girl_audio.mp3" \
    --video_out_path "new_high_vid.mp4"