import os
import open_clip
from torchvision import datasets
from torchvision import transforms as T

from engine.datasets import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def get_model(name: str):
    name = name.lower()
    if name=='vit_b':
        model,_,_ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    elif name=='vit_l':
        model,_,_ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    elif name=='t5_small':
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small", model_max_length=128)
        transformer = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
        model = (tokenizer, transformer)
    elif name=='t5_base':
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base", model_max_length=128)
        transformer = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
        model = (tokenizer, transformer)
    elif name=='t5_large':
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large", model_max_length=128)
        transformer = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-large")
        model = (tokenizer, transformer)
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
        train_dst = CUB200(data_root, split='train', download=False, transform=train_transform)
        val_dst = CUB200(data_root, split='test', download=False, transform=val_transform)
        classes_name = val_dst.classes
    elif name=='cars':
        num_classes = 196
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
        train_dst = StanfordCars(data_root, split='train', download=False, transform=train_transform)
        val_dst = StanfordCars(data_root, split='test', download=False, transform=val_transform)
        idx_to_class = dict((v, k) for k, v in train_dst.class_to_idx.items())
        classes_name = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]
    elif name=='resisc45':
        num_classes = 45
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
        train_dst = RESISC45(data_root, split='train', transforms=train_transform)
        val_dst = RESISC45(data_root, split='test', transforms=val_transform)
        classes_name = ['satellite imagery of ' + ' '.join(c.split('_')) for c in RESISC45.classes]
    elif name=='eurosat':
        num_classes = 10
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
        data_root = os.path.join( data_root, 'EuroSAT' )
        train_dst = datasets.ImageFolder(os.path.join( data_root, 'train' ), transform=train_transform)
        val_dst = datasets.ImageFolder(os.path.join( data_root, 'test' ), transform=val_transform)
        classes_name = ['annual crop land', 'forest', 'brushland or shrubland', 'highway or road',
                        'industrial buildings or commercial buildings', 'pasture land', 'permanent crop land',
                        'residential buildings or homes or apartments', 'river', 'lake or sea']
        classes_name = ['satellite imagery of ' + i for i in classes_name]
    elif name=='svhn':
        num_classes = 10
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
        data_root = os.path.join( data_root, 'torchdata' )
        train_dst = datasets.SVHN(data_root, split='train', download=False, transform=train_transform)
        val_dst = datasets.SVHN(data_root, split='test', download=False, transform=val_transform)
        classes_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        classes_name = [f'number: "{c}"' for c in classes_name]
    elif name=='mnist':
        num_classes = 10
        train_transform = T.Compose([
            T.Grayscale(3),
            T.Resize(224),
            T.RandomCrop(224, padding=16),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = T.Compose([
            T.Grayscale(3),
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        data_root = os.path.join( data_root, 'torchdata' )
        train_dst = datasets.MNIST(data_root, train=True, download=False, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=False, transform=val_transform)
        classes_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        classes_name = [f'number: "{c}"' for c in classes_name]
    elif name=='gtsrb':
        num_classes = 43
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
        train_dst = GTSRB(data_root, split='train', download=False, transform=train_transform)
        val_dst = GTSRB(data_root, split='test', download=False, transform=val_transform)
        classes_name = val_dst.classnames
        classes_name = [f'"{c}" traffic sign' for c in classes_name]
    elif name=='sun397':
        num_classes = 362
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
        data_root = os.path.join( data_root, 'sun397' )
        train_dst = datasets.ImageFolder(os.path.join( data_root, 'train' ), transform=train_transform)
        val_dst = datasets.ImageFolder(os.path.join( data_root, 'test' ), transform=val_transform)
        idx_to_class = dict((v, k) for k, v in train_dst.class_to_idx.items())
        classes_name = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]
    elif name=='dtd':
        num_classes = 47
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
        data_root = os.path.join( data_root, 'dtd' )
        train_dst = datasets.ImageFolder(os.path.join( data_root, 'train' ), transform=train_transform)
        val_dst = datasets.ImageFolder(os.path.join( data_root, 'test' ), transform=val_transform)
        idx_to_class = dict((v, k)for k, v in train_dst.class_to_idx.items())
        classes_name = [idx_to_class[i].replace('_', ' ')+' texture' for i in range(len(idx_to_class))]
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