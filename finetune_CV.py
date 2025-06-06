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
parser.add_argument('--model', default='vit_b')
parser.add_argument('--dataset', default='cifar100')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--batches', default=3000, type=int, metavar='N',
                    help='number of total batches to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float,
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
    model = registry.get_model(args.model)
    model = model.to(device)

    num_classes, classes_name, train_dataset, val_dataset = registry.get_dataset(args.dataset, args.data_root)
    clip_features = load_clip_features(classes_name, model, device=device)
    # print(train_dataset)
    # print(val_dataset)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.batches)

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
        val_acc, val_loss = validate_CV(model, val_iter, clip_features, data_scale=1.0)
        print('Model={}_{}.pth, Val Loss={:.4f}, Val Acc={:.4f}'
              .format(args.dataset, args.model, val_loss, val_acc))
        return

    ############################################
    # Train Loop
    ############################################
    best_acc = 0
    losses = []
    total = correct = 0
    for i in range(1, args.batches+1):
        model.train()

        optimizer.zero_grad(set_to_none=True)
        images, target = train_iter.next()
        with autocast():
            images, target = images.to(device), target.to(device)
            encodings = model.encode_image(images)
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

        if i % 100 == 0:
            train_acc = correct / total
            train_loss = np.mean(losses)
            losses = []
            total = correct = 0

            val_acc, val_loss = validate_CV(model, val_iter, clip_features, data_scale=1.0)

            torch.save(model.state_dict(), last_ckpt)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_ckpt)

            print('Batch={:d}, Train Loss={:.4f}, Train Acc={:.4f}, Val Loss={:.4f}, Val Acc={:.4f}, Lr={:.6f}'
                  .format(i, train_loss, train_acc, val_loss, val_acc, optimizer.param_groups[0]['lr']))

    print("Best Acc: %.4f" % best_acc)
    

if __name__ == '__main__':
    main()