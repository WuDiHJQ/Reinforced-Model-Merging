import os
import torch
import argparse
import registry
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from engine.datasets.utils import DatasetWrapper
from engine.utils import DataIter, reset_random, validate_NLP, compute_forward_loss


parser = argparse.ArgumentParser(description='NLP Training')
parser.add_argument('--data_root', default='../data')
parser.add_argument('--model', default='t5_base')
parser.add_argument('--dataset', default='qasc')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--batches', default=1000, type=int, metavar='N',
                    help='number of total batches to run')
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
    (tokenizer, transformer) = registry.get_model(args.model)
    transformer = transformer.to(device)
    
    ############################################
    # Setup dataset
    ############################################
    _,_, train_dst, val_dst = registry.get_dataset(args.dataset, args.data_root)
    train_dst = DatasetWrapper(train_dst, tokenizer, device)  
    val_dst = DatasetWrapper(val_dst, tokenizer, device)
    
    train_iter = DataIter(torch.utils.data.DataLoader(
                          train_dst, batch_size=args.batch_size,
                          num_workers=args.workers, shuffle=True,
                          collate_fn=train_dst.collate_fn))
    
    val_iter = DataIter(torch.utils.data.DataLoader(
                        val_dst, batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False,
                        collate_fn=val_dst.collate_fn))
    
    ############################################
    # Setup optimizer
    ############################################   
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(transformer.parameters(), args.lr, weight_decay=0, eps=1e-8)

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
        transformer.load_state_dict(torch.load(best_ckpt))
        val_acc = validate_NLP(transformer, val_iter, data_scale=1.0)
        print('Model={}_{}.pth, Val Acc={:.4f}'.format(args.dataset, args.model, val_acc))
        return

    ############################################
    # Train Loop
    ############################################
    best_acc = 0
    losses = []
    for i in range(1, args.batches+1):
        transformer.train()

        batch = train_iter.next()
        with autocast(dtype=torch.bfloat16):
            loss = compute_forward_loss(transformer, batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        losses.append(loss.item())
        
        if i % 100 == 0:
            val_acc = validate_NLP(transformer, val_iter, data_scale=1.0)

            # torch.save(transformer.state_dict(), last_ckpt)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(transformer.state_dict(), best_ckpt)

            print('Batch={:d}, Train Loss={:.4f}, Val Acc={:.4f}'
                  .format(i, np.mean(losses), val_acc))
            losses = []


if __name__ == '__main__':
    main()