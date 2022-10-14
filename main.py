import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import models as model_e
import generator as model_g
import task_generator as loader
import numpy as np
import random
import scipy.stats as stats
from utils import  AverageMeter, accuracy, mkdir_p

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/data/miniImageNet/',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', default=0, type=int,
                    metavar='E', help='evaluate model on validation set')
parser.add_argument('--N_reference', default=48, type=int,
                    metavar='R', help='N_reference (default: 64)')
parser.add_argument('--N_way', default=5, type=int,
                    metavar='NWAY', help='N_way (default: 5)')
parser.add_argument('--N_shot', default=1, type=int,
                    metavar='NSHOT', help='N_shot (default: 1)')
parser.add_argument('--N_query', default=15, type=int,
                    metavar='NQUERY', help='N_query (default: 15)')
parser.add_argument('--gpu', default='0')
parser.add_argument('--diversity_parmater', default=5.0, type=float, metavar='M',
                    help='diversity_parmater')
parser.add_argument('--kl_parmater', default=1.0, type=float, metavar='M',
                    help='kl_parmater')
parser.add_argument('--N_generate', default=64, type=int, metavar='M',
                    help='N_generate')
parser.add_argument('--N_reference_per_class', default=2, type=int,
                    metavar='R', help='N_reference_per_class (default: 2)')

SEED = 3
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def main():
    global args
    args = parser.parse_args()
    set_gpu(args.gpu)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    model_E = model_e.Net().cuda()
    model_G = model_g.GeneratorNet(N_generate=args.N_generate).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD([
        {'params': model_E.parameters()},
        {'params': model_G.parameters(), 'lr': args.lr}],
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,25,30,40,45,50,55,60,65,80,100,120,140], gamma=0.4)
    # Data loading code
    mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
    std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
    normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
    
    train_aug_dataset = loader.ImageLoader(
        args.data,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=16),
            transforms.RandomHorizontalFlip(),

            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize,
        ]), is_train=True)

    base_train_loader = torch.utils.data.DataLoader(
        train_aug_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    gen_support_loader = torch.utils.data.DataLoader(
        train_aug_dataset, batch_size=args.N_reference*args.N_reference_per_class, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler = loader.GeneratorSupportSampler(n_class=64, n_support_pairs=args.N_reference, N_reference_per_class = args.N_reference_per_class))

    gen_train_loader = torch.utils.data.DataLoader(
        train_aug_dataset, batch_size=args.N_way*(args.N_query+args.N_shot), shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler = loader.GeneratorSampler(num_of_class=args.N_way, num_per_class=args.N_query+args.N_shot, n_class=64))
    
    gen_support_for_val_loader = torch.utils.data.DataLoader(
        train_aug_dataset, batch_size=args.N_reference * args.N_reference_per_class, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=loader.GeneratorSupportSampler(n_class=64, n_support_pairs=args.N_reference, N_reference_per_class = args.N_reference_per_class))

    gen_support_for_test_loader = torch.utils.data.DataLoader(
        train_aug_dataset, batch_size=args.N_reference * args.N_reference_per_class, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=loader.GeneratorSupportSampler(n_class=64, n_support_pairs=args.N_reference, N_reference_per_class = args.N_reference_per_class))
   
    val_dataset = loader.ImageLoader(
        args.data,
        transforms.Compose([
            transforms.Resize((224, 224)),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize,
        ]), is_val=True)

    gen_val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.N_way * (args.N_query + args.N_shot), shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=loader.GeneratorSampler(num_of_class=args.N_way, num_per_class=args.N_query + args.N_shot, n_class=16))
    
    test_dataset = loader.ImageLoader(
        args.data,
        transforms.Compose([
            transforms.Resize((224, 224)),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize,
        ]), is_test=True)

    gen_test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot), shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=loader.GeneratorSampler(num_of_class=args.N_way, num_per_class=args.N_query + args.N_shot, n_class=20))
    


    if args.evaluate:
        if args.resume:
            print('==> Resuming from generator checkpoint..')
            assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(args.resume)
            model_G.load_state_dict(checkpoint['state_dict_G'])
            model_E.load_state_dict(checkpoint['state_dict_E'])
        losses = AverageMeter()
        top1 = AverageMeter()
        H = AverageMeter()
        for i in range(1):
            val_loss, val_acc, h = validate(gen_support_for_test_loader, gen_test_loader, model_E, model_G, criterion)
            losses.update(val_loss)
            top1.update(val_acc)
            H.update(h)
            print(i, losses.avg, top1.avg, H.avg)
        return 0

    train_acc, train_loss, base_train_loss, base_train_acc = 0, 0, 0, 0
    best_acc = 0
    TRAIN_PHASE = ['a']*30 + ['m']*20 + (['a']*1 + ['m']*4)*100 
    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
        train_phase = TRAIN_PHASE[epoch]
        if train_phase == 'a':
            print('phase: base_train...')
            base_train_loss, base_train_acc = base_train(base_train_loader, model_E, criterion, optimizer, epoch)
        else:
            print('phase: meta_train...')
            train_loss, train_acc = train(gen_support_loader, gen_train_loader, model_E, model_G, criterion, optimizer, epoch)
            if epoch > 31:
                _, val_acc, _ = validate(gen_support_for_val_loader, gen_val_loader, model_E, model_G, criterion)
                print('current epoch val acc: {:}'.format(val_acc))
                _, test_acc, _ = validate(gen_support_for_test_loader, gen_test_loader, model_E, model_G, criterion)
                print('current epoch test acc: {:}'.format(test_acc))
                
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_epoch = epoch
                    save_checkpoint({
                    'epoch': epoch,
                    'state_dict_E': model_E.state_dict(),
                    'state_dict_G': model_G.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                    }, epoch, checkpoint=args.checkpoint)
    print('best_test_acc:',best_acc)
    print('best_test_epoch:',best_epoch)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m = np.mean(a)
    se = stats.sem(a)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m,h


def train(gen_support_loader, gen_train_loader, model_E, model_G, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()  
    model_E.train()
    model_G.train()
    for inter in range(400):
        input, target = gen_train_loader.__iter__().next()
        gen_support_input, gen_support_target = gen_support_loader.__iter__().next()
        input = input.cuda()
        support_input = input.view(args.N_way, args.N_query + args.N_shot, 3, 224, 224)[:,-args.N_shot:,:,:,:].contiguous().view(-1, 3, 224, 224)
        query_input   = input.view(args.N_way, args.N_query + args.N_shot, 3, 224, 224)[:,:-args.N_shot,:,:,:].contiguous().view(-1, 3, 224, 224)
        gen_support = gen_support_input.cuda()
        support_input, _ = model_E(support_input) # (5,1024)
        support_input = support_input.view(args.N_way, args.N_shot, -1)
        # 第一种方法：对shot维度取mean
        support_input = torch.mean(support_input, 1)
        gen_support, _ = model_E(gen_support) # (64,1024)
        gen_support = gen_support.view(args.N_reference, args.N_reference_per_class, -1)
        query_input  , _ = model_E(query_input)  # (75,1024)
        weight, diversity_loss, kl_loss = model_G(support_input, gen_support) # (5,1024)
        weight = model_E.l2_norm(weight)
        predict = torch.matmul(query_input, torch.transpose(weight,0,1))*model_G.s
        gt = np.tile(range(args.N_way), args.N_query)
        gt.sort()
        gt = torch.cuda.LongTensor(gt)
        acc = (predict.topk(1)[1].view(-1)==gt).float().sum(0)/gt.shape[0]*100.
        loss = criterion(predict, gt) + args.diversity_parmater*diversity_loss + args.kl_parmater*kl_loss
        losses.update(loss.item(), predict.size(0))
        top1.update(acc.item(), predict.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (losses.avg, top1.avg)

def validate(gen_support_for_test_loader, gen_test_loader, model_E, model_G, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    model_E.eval()
    model_G.eval()
    with torch.no_grad():
        accuracies = []
        for inter in range(600):
            input, target = gen_test_loader.__iter__().next()
            gen_support_input, gen_support_target = gen_support_for_test_loader.__iter__().next()
            support_input = input.view(args.N_way,args.N_query+args.N_shot,3,224,224)[:,-args.N_shot:,:,:,:].contiguous().view(-1, 3, 224, 224).cuda()
            query_input = input.view(args.N_way,args.N_query+args.N_shot,3,224,224)[:,:-args.N_shot,:,:,:].contiguous().view(-1, 3, 224, 224).cuda()
            gen_support = gen_support_input.cuda()
            support_input, _ = model_E(support_input)
            support_input = support_input.view(args.N_way, args.N_shot, -1)
            # 第一种方法：对shot维度取mean
            support_input = torch.mean(support_input, 1)
            gen_support, _ = model_E(gen_support) # (64,1024)
            gen_support = gen_support.view(args.N_reference, args.N_reference_per_class, -1)
            query_input, _ = model_E(query_input)
            weight, diversity_loss, kl_loss = model_G(support_input, gen_support) # (5,1024)
            weight = model_E.l2_norm(weight)
            predict = torch.matmul(query_input, torch.transpose(weight,0,1))*model_G.s
            gt = np.tile(range(args.N_way), args.N_query)
            gt.sort()
            gt = torch.cuda.LongTensor(gt)
            acc = (predict.topk(1)[1].view(-1)==gt).float().sum(0)/gt.shape[0]*100.
            accuracies.append(acc.item())
            loss = criterion(predict, gt) + args.diversity_parmater*diversity_loss + args.kl_parmater*kl_loss
            losses.update(loss.item(), predict.size(0))
            top1.update(acc.item(), predict.size(0))
    mean, h = mean_confidence_interval(accuracies)
    return (losses.avg, top1.avg, h)

def base_train(base_train_loader, model_E, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    model_E.train()
    for batch_idx, (input, target) in enumerate(base_train_loader):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        _, output = model_E(input)
        loss = criterion(output, target)
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (losses.avg, top1.avg)

def save_checkpoint(state, epoch, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)
    torch.save(state, filepath)
    print('save checkpoint success', epoch)

if __name__ == '__main__':
    main()
