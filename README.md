## YOLO v3: Backbone, Head, Detect


Train the model
```commandline
python -m torch.distributed.run --nproc_per_node 1 train.py --epochs 300 --batch-size 32 --weight "" --linear-lr

```


Evaluate (Reproduce the results):
```commandline
python val.py --weights weights/yolov3-tiny.pt
```

Results of YOLOv3 Tiny model:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.169
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.339
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.149
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.078
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.210
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.224
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.173
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.357
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.185
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.425
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.476

```




| Model                                        | Num. Params |
|----------------------------------------------|:-----------:|
| [YOLOv3](./yolov3/models/yolov3.py)          |   61.95M    | 
| [YOLOv3 Tiny](./yolov3/models/yolov3tiny.py) |    8.85M    |  
| [YOLOv3 SPP](./yolov3/models/yolov3spp.py)   |     63M     | 

## Reference:

- Ultralytics
