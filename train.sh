python -m torch.distributed.run --nproc_per_node 1 train.py --epochs 200 --weights "weights/weights/last.pt"
