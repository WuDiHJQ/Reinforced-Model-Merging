import argparse
import os
import random
import registry
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.cuda.amp import autocast, GradScaler
from engine.utils import DataIter, load_clip_features, reset_random, validate_CV


parser = argparse.ArgumentParser(description='Vision Training')
parser.add_argument('--data_root', default='../data')
parser.add_argument('--model', default='vit_s')
parser.add_argument('--dataset', default='cifar100')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')


def main():
    args = parser.parse_args()
    reset_random(args.seed)
    
    print('='*100)
    for k, v in vars(args).items():
        print("%s: %s"%(k,v))
    print('='*100)
    
    main_worker(args)

    
def main_worker(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ############################################
    # Setup models
    ############################################
    num_classes, classes_name, train_dataset, val_dataset = registry.get_dataset(args.dataset, args.data_root)
    clip_features = load_clip_features(classes_name, device=device)
    # print(train_dataset)
    # print(val_dataset)
    
    model = registry.get_model(args.model, num_classes=512, pretrained=True)
    model = model.to(device)

    ############################################
    # Setup dataset
    ############################################
    cudnn.benchmark = True
    train_iter = DataIter(torch.utils.data.DataLoader(
                          train_dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers))
    val_iter = DataIter(torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers))

    ############################################
    # Setup optimizer
    ############################################
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
    
    ne_iters = len(train_iter.dataloader)
    lr_schedule = np.interp(np.arange(1 + args.epochs * ne_iters),
                            [0, 5 * ne_iters, args.epochs * ne_iters], [0, 1, 0])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
    
    ############################################
    # Setup Folder
    ############################################
    path = 'checkpoint/%s_%s'%(args.dataset, args.model)
    if not os.path.exists(path):
        os.makedirs(path)
    best_ckpt = path + '/best.pth'
    last_ckpt = path + '/last.pth'
    
    ############################################
    # Evaluate
    ############################################
    if args.evaluate:
        model.load_state_dict(torch.load(best_ckpt))
        val_acc, val_loss = validate_CV(model, val_iter, clip_features, None, data_scale=1.0)
        print('Model={}_{}.pth, Val Loss={:.4f}, Val Acc={:.4f}'
              .format(args.dataset, args.model, val_loss, val_acc))
        return

    ############################################
    # Train Loop
    ############################################
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        correct = 0
        total = 0
        for i, (images, target) in enumerate(tqdm(train_iter.dataloader)):
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                images, target = images.to(device), target.to(device)
                encodings = model(images)
                normed_encodings = encodings / encodings.norm(dim=-1, keepdim=True)
                logits = (100.0 * normed_encodings @ clip_features.T)
                pred = logits.argmax(dim=1)
                loss = criterion(logits, target)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            losses.append(loss.item())
            
            for gt, p in zip(target, pred):
                is_correct = (gt == p).item()
                correct += is_correct

            total += images.shape[0]
            
        train_acc = correct / total
        train_loss = np.mean(losses)
            
        val_acc, val_loss = validate_CV(model, val_iter, clip_features, None, data_scale=1.0)
        
        torch.save(model.state_dict(), last_ckpt)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)
            
        print('Epoch={:d}, Train Loss={:.4f}, Train Acc={:.4f}, Val Loss={:.4f}, Val Acc={:.4f}, Lr={:.6f}'
              .format(epoch, train_loss, train_acc, val_loss, val_acc, optimizer.param_groups[0]['lr']))
        
    print("Best Acc: %.4f" % best_acc)
    

if __name__ == '__main__':
    main()