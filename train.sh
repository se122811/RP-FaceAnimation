# train 파일
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12356 train.py --config ./config/face.yaml --name face_3d
