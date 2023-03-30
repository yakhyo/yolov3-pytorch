python -m torch.distributed.run --nproc_per_node 1 train.py --epochs 500 --batch-size 64 --weight "weights/last.pt"
