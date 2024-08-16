from torch import nn
import timm
from .vit_clip import *

def vit_s_timm(num_classes=0, pretrained=True):

    net = timm.create_model("hf_hub:timm/vit_small_patch32_224.augreg_in21k_ft_in1k", pretrained=pretrained)
    if num_classes==0:
        net.head = nn.Identity()
    else:
        net.head = nn.Linear(net.head.in_features, num_classes)
    return net

def vit_b_timm(num_classes=0,pretrained=True):
    
    net = timm.create_model("hf_hub:timm/vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=pretrained)
    if num_classes==0:
        net.head = nn.Identity()
    else:
        net.head = nn.Linear(net.head.in_features, num_classes)
    return net