# Reinforced Model Merging
Code for paper "Reinforced Model Merging".


<div align="center">
<img src="assets/RMM.png" width="100%"></img> 
</div>


## Quick Start

Here we provide a quick start for RMM.

All you need to do is **Train** the models and **Merge** them as you wish.

### Merge ViT-B/16 on CUB-200 & Stanford Dogs datasets

#### 1. Fine-tune Models 
```bash
python finetune_CV.py --model vit_b --dataset cub
python finetune_CV.py --model vit_b --dataset dogs
```

#### 2. Merge Models
* choose merging method [ties, dare, dare_ties]
* set smaller data_scale for faster merging
```bash
python train_RMM.py --model vit_b --method ties --dataset cub,dogs --data_scale 0.1
```

#### 3. Test Merged Model
```bash
python test_RMM.py --model vit_b --method ties --dataset cub,dogs
```