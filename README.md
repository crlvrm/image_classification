# image_classification
基于PyTorch集成Mobilenet、ShuffleNet、ResNet等网络的图像分类demo

## 数据集格式
<p>data</p>
<p>-train</p>
<p>--cls1</p>
<p>---1.jpg</p>
<p>---...</p>
<p>--cls2</p>
<p>-val</p>
<p>--cls1</p>
<p>--cls2</p>


## 配置文件config/complete.yaml
集成了所有数据增强和训练策略的接口，其中model只能调用torchvision中已构建的模型，详细见文件中的注释

## 训练
### 单卡训练

> python train.py --cfgs './config/complete.yaml'

### 多卡训练
> CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 main.py --cfgs './config/complete.yaml'

## 单张推理
推理参数需保持和训练参数一样
> python train.py --img './img.jpg' --pt './best.pt' --num_classes 7
