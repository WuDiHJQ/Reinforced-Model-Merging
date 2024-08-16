# OOD-KA
Code for paper "Reinforced Model Merging"


<div align="center">
<img src="assets/RMM.png" width="80%"></img> 
</div>

## Requirements

To install requirements:
```bash
pip install -r requirements.txt
```

## Reproduce our results

### Image Classification

#### 1. Split CIFAR100
```bash
cd engine/datasets
python split_cifar100.py
```

#### 2. Train Teachers
```bash
python train_scratch.py --lr 0.1 --batch-size 128 --model wrn16_2 --dataset cifar100_part0 --gpu 0
python train_scratch.py --lr 0.1 --batch-size 128 --model wrn16_2 --dataset cifar100_part1 --gpu 0
```

#### 3. OOD-KA
```bash
python OOD_KA.py \
--model wrn16_2 \
--unlabeled cifar10 \
--lr 1e-3 \
--lr_g 1e-3 \
--z_dim 100 \
--oh 1.0 \
--bn 1.0 \
--local 1.0 \
--adv 1.0 \
--sim 1.0 \
--balance 10.0
```

```
<div align="center">
<img src="assets/results.png" width="80%"></img> 
</div>
```

### Semantic Segmentation

#### 1. Train Teachers with DeepLabV3
```bash
python train_deeplab.py --model deeplabv3plus_mobilenet --dataset voc --output_stride 16 --lr 0.01 --batch_size 16
python train_deeplab.py --model deeplabv3plus_mobilenet --dataset nyu --output_stride 16 --lr 0.01 --batch_size 16
```

#### 1. OOD-KA for semantic segmentation
```bash
python PFA_deeplab.py \
--model deeplabv3plus_mobilenet \
--output_stride 16 \
--batch_size 16 \
--lr 1e-2 \
--lr_g 1e-3 \
--z_dim 256 \
--oh 1.0 \
--bn 0.5 \
--local 0.5 \
--adv 1.0 \
--sim 1.0 \
--balance 1.0 
```
