CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 12391 inference.py --config ./config/face_demo.yaml --name face_uv_with_source --no_resume --output_dir ./vox_result/face_reenactment_cross