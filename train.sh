python -m torch.distributed.run --nproc_per_node 1 train.py --epochs 300 --batch-size 32 --weight "weights/last.pt"
