import os
from torchvision import datasets
from torchvision import transforms as T

from engine.models import vit, t5
from engine.datasets import *


def get_model(name: str, num_classes=512, pretrained=True):
    name = name.lower()
    if name=='vit_s':
        model = vit.vit_s_timm(num_classes, pretrained)
    elif name=='vit_b':
        model = vit.vit_b_timm(num_classes, pretrained)
    elif name=='t5_small':
        model = t5.t5_small()
    elif name=='t5_base':
        model = t5.t5_base()
    elif name=='t5_large':
        model = t5.t5_large()
    return model


def get_dataset(name: str, data_root: str='data'):
    name = name.lower()
    data_root = os.path.expanduser( data_root )
    if name=='cifar10':
        num_classes = 10
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.Resize(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = T.Compose([
            T.Resize(224), 
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
        data_root = os.path.join( data_root, 'torchdata' )
        train_dst = datasets.CIFAR10(data_root, train=True, download=False, transform=train_transform)
        val_dst = datasets.CIFAR10(data_root, train=False, download=False, transform=val_transform)
        classes_name = val_dst.classes
    elif name=='cifar100':
        num_classes = 100
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.Resize(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = T.Compose([
            T.Resize(224), 
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
        data_root = os.path.join( data_root, 'torchdata' )
        train_dst = datasets.CIFAR100(data_root, train=True, download=False, transform=train_transform)
        val_dst = datasets.CIFAR100(data_root, train=False, download=False, transform=val_transform)
        classes_name = val_dst.classes
    elif name=='dogs':
        num_classes = 120
        train_transform = T.Compose([
            T.Resize(224),
            T.RandomCrop(224, padding=16),
            T.RandomHorizontalFlip(),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
        data_root = os.path.join( data_root, 'Stanford_Dogs' )
        train_dst = StanfordDogs(data_root, split='train', download=False, transform=train_transform)
        val_dst = StanfordDogs(data_root, split='test', download=False, transform=val_transform)
        classes_name = val_dst.classes
    elif name=='cub':
        num_classes = 200
        train_transform = T.Compose([
            T.Resize(224),
            T.RandomCrop(224, padding=16),
            T.RandomHorizontalFlip(),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
        # data_root = os.path.join( data_root, 'CUB_200_2011' )
        train_dst = CUB200(data_root, split='train', download=False, transform=train_transform)
        val_dst = CUB200(data_root, split='test', download=False, transform=val_transform)
        classes_name = val_dst.classes
    elif name=='cifar10_half0':
        num_classes = 5
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.Resize(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = T.Compose([
            T.Resize(224), 
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
        data_root = os.path.join( data_root, 'torchdata' )
        train_dst = Split_Dataset(datasets.CIFAR10(data_root, train=True, download=False, transform=train_transform),
                                  10, range(0,5), '%s/%s_%s.pkl'%(data_root,name,'train'))
        val_dst = Split_Dataset(datasets.CIFAR10(data_root, train=False, download=False, transform=val_transform),
                                10, range(0,5), '%s/%s_%s.pkl'%(data_root,name,'test'))
        classes_name = val_dst.dataset.classes[:5]
    elif name=='cifar10_half1':
        num_classes = 5
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.Resize(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = T.Compose([
            T.Resize(224), 
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
        data_root = os.path.join( data_root, 'torchdata' )
        train_dst = Split_Dataset(datasets.CIFAR10(data_root, train=True, download=False, transform=train_transform),
                                  10, range(5,10), '%s/%s_%s.pkl'%(data_root,name,'train'))
        val_dst = Split_Dataset(datasets.CIFAR10(data_root, train=False, download=False, transform=val_transform),
                                10, range(5,10), '%s/%s_%s.pkl'%(data_root,name,'test'))
        classes_name = val_dst.dataset.classes[5:]
    elif name=='cifar100_half0':
        num_classes = 50
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.Resize(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = T.Compose([
            T.Resize(224), 
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
        data_root = os.path.join( data_root, 'torchdata' )
        train_dst = Split_Dataset(datasets.CIFAR100(data_root, train=True, download=False, transform=train_transform),
                                  100, range(0,50), '%s/%s_%s.pkl'%(data_root,name,'train'))
        val_dst = Split_Dataset(datasets.CIFAR100(data_root, train=False, download=False, transform=val_transform),
                                100, range(0,50), '%s/%s_%s.pkl'%(data_root,name,'test'))
        classes_name = val_dst.dataset.classes[:50]
    elif name=='cifar100_half1':
        num_classes = 50
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.Resize(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = T.Compose([
            T.Resize(224), 
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
        data_root = os.path.join( data_root, 'torchdata' )
        train_dst = Split_Dataset(datasets.CIFAR100(data_root, train=True, download=False, transform=train_transform),
                                  100, range(50,100), '%s/%s_%s.pkl'%(data_root,name,'train'))
        val_dst = Split_Dataset(datasets.CIFAR100(data_root, train=False, download=False, transform=val_transform),
                                100, range(50,100), '%s/%s_%s.pkl'%(data_root,name,'test'))
        classes_name = val_dst.dataset.classes[50:]
    elif name=='cub_half0':
        num_classes = 100
        train_transform = T.Compose([
            T.Resize(224),
            T.RandomCrop(224, padding=16),
            T.RandomHorizontalFlip(),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
        # data_root = os.path.join( data_root, 'CUB_200_2011' )
        train_dst = Split_Dataset(CUB200(data_root, split='train', download=False, transform=train_transform), 
                                  200, range(0,100), '%s/%s_%s.pkl'%(data_root,name,'train'))
        val_dst = Split_Dataset(CUB200(data_root, split='test', download=False, transform=val_transform), 
                                200, range(0,100), '%s/%s_%s.pkl'%(data_root,name,'test'))
        classes_name = val_dst.dataset.classes[:100]
    elif name=='cub_half1':
        num_classes = 100
        train_transform = T.Compose([
            T.Resize(224),
            T.RandomCrop(224, padding=16),
            T.RandomHorizontalFlip(),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
        # data_root = os.path.join( data_root, 'CUB_200_2011' )
        train_dst = Split_Dataset(CUB200(data_root, split='train', download=False, transform=train_transform), 
                                  200, range(100,200), '%s/%s_%s.pkl'%(data_root,name,'train'))
        val_dst = Split_Dataset(CUB200(data_root, split='test', download=False, transform=val_transform), 
                                200, range(100,200), '%s/%s_%s.pkl'%(data_root,name,'test'))
        classes_name = val_dst.dataset.classes[100:]
    elif name=='dogs_half0':
        num_classes = 60
        train_transform = T.Compose([
            T.Resize(224),
            T.RandomCrop(224, padding=16),
            T.RandomHorizontalFlip(),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
        data_root = os.path.join( data_root, 'Stanford_Dogs' )
        train_dst = Split_Dataset(StanfordDogs(data_root, split='train', download=False, transform=train_transform),
                                  120, range(0,60), '%s/%s_%s.pkl'%(data_root,name,'train'))
        val_dst = Split_Dataset(StanfordDogs(data_root, split='test', download=False, transform=val_transform),
                                120, range(0,60), '%s/%s_%s.pkl'%(data_root,name,'test'))
        classes_name = val_dst.dataset.classes[:60]
    elif name=='dogs_half1':
        num_classes = 60
        train_transform = T.Compose([
            T.Resize(224),
            T.RandomCrop(224, padding=16),
            T.RandomHorizontalFlip(),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
        data_root = os.path.join( data_root, 'Stanford_Dogs' )
        train_dst = Split_Dataset(StanfordDogs(data_root, split='train', download=False, transform=train_transform),
                                  120, range(60,120), '%s/%s_%s.pkl'%(data_root,name,'train'))
        val_dst = Split_Dataset(StanfordDogs(data_root, split='test', download=False, transform=val_transform),
                                120, range(60,120), '%s/%s_%s.pkl'%(data_root,name,'test'))
        classes_name = val_dst.dataset.classes[60:]
    elif name=='qasc':
        num_classes = classes_name = None
        dst_reader = QASC()
        train_dst = dst_reader.get_dataset("train", 0, is_evaluation=False)
        val_dst = dst_reader.get_dataset('validation', 0, is_evaluation=True)
    elif name=='wiki_qa':
        num_classes = classes_name = None
        dst_reader = WikiQA()
        train_dst = dst_reader.get_dataset("train", 0, is_evaluation=False)
        val_dst = dst_reader.get_dataset('validation', 0, is_evaluation=True)
    elif name=='quartz':
        num_classes = classes_name = None
        dst_reader = QuaRTz()
        train_dst = dst_reader.get_dataset("train", 0, is_evaluation=False)
        val_dst = dst_reader.get_dataset('validation', 0, is_evaluation=True)
    elif name=='paws':
        num_classes = classes_name = None
        dst_reader = PAWS()
        train_dst = dst_reader.get_dataset("train", 0, is_evaluation=False)
        val_dst = dst_reader.get_dataset('validation', 0, is_evaluation=True)
    elif name=='story_cloze':
        num_classes = classes_name = None
        dst_reader = StoryCloze()
        train_dst = dst_reader.get_dataset("train", 0, is_evaluation=False)
        val_dst = dst_reader.get_dataset('validation', 0, is_evaluation=True)
    elif name=='winogrande':
        num_classes = classes_name = None
        dst_reader = Winogrande()
        train_dst = dst_reader.get_dataset("train", 0, is_evaluation=False)
        val_dst = dst_reader.get_dataset('validation', 0, is_evaluation=True)
    elif name=='wsc':
        num_classes = classes_name = None
        dst_reader = WSC()
        train_dst = dst_reader.get_dataset("train", 0, is_evaluation=False)
        val_dst = dst_reader.get_dataset('validation', 0, is_evaluation=True)
    else:
        raise NotImplementedError

    return num_classes, classes_name, train_dst, val_dst