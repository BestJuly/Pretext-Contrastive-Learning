"""Finetune 3D CNN."""
import os
import argparse
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.optim as optim
# uncomment because tensorboard does not support torch 1.4.0
#from tensorboardX import SummaryWriter

from datasets.ucf101 import UCF101Dataset
from datasets.hmdb51 import HMDB51Dataset
from models.r3d import R3DNet

import ast

def load_pretrained_weights(ckpt_path):
    """load pretrained weights and adjust params name."""
    adjusted_weights = {}
    pretrained_weights = torch.load(ckpt_path)
    for name, params in pretrained_weights.items():
        if 'base_network' in name:
            name = name[name.find('.')+1:]
            adjusted_weights[name] = params
            print('Pretrained weight name: [{}]'.format(name))
    return adjusted_weights


def diff(x):
    shift_x = torch.roll(x,1,2)
    return shift_x - x
    #return ((shift_x -x) + 1)/2


def train(args, model, criterion, optimizer, device, train_dataloader, writer, epoch):
    torch.set_grad_enabled(True)
    model.train()

    running_loss = 0.0
    correct = 0
    for i, data in enumerate(train_dataloader, 1):
        # get inputs
        clips, idxs = data
        inputs = clips.to(device)
        targets = idxs.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward and backward
        if args.modality == 'res':
            outputs = model(diff(inputs))
        else:
            outputs = model(inputs)
        #outputs = model(inputs) # return logits here
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # compute loss and acc
        running_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print statistics and write summary every N batch
        if i % args.pf == 0:
            avg_loss = running_loss / args.pf
            avg_acc = correct / (args.pf * args.bs)
            print('Train epoch: [{0:3d}/{1:3d}][{2:4d}/{3:4d}]\t'
                'loss: {4:.3f}, acc: {5:.3f}, lr: {lr}'.format(
                epoch, args.epochs, i + 1, len(train_dataloader),
                avg_loss, avg_acc, lr=optimizer.param_groups[0]['lr']), end='\r')
            step = (epoch-1)*len(train_dataloader) + i
            #writer.add_scalar('train/CrossEntropyLoss', avg_loss, step)
            #writer.add_scalar('train/Accuracy', avg_acc, step)
            running_loss = 0.0
            correct = 0
    print('')


def validate(args, model, criterion, device, val_dataloader, writer, epoch):
    torch.set_grad_enabled(False)
    model.eval()
    
    total_loss = 0.0
    correct = 0
    for i, data in enumerate(val_dataloader):
        # get inputs
        clips, idxs = data
        inputs = clips.to(device)
        targets = idxs.to(device)
        # forward
        if args.modality == 'res':
            outputs = model(diff(inputs))
        else:
            outputs = model(inputs)
        #outputs = model(inputs) # return logits here
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    avg_loss = total_loss / len(val_dataloader)
    avg_acc = correct / len(val_dataloader.dataset)
    #writer.add_scalar('val/CrossEntropyLoss', avg_loss, epoch)
    #writer.add_scalar('val/Accuracy', avg_acc, epoch)
    print('[VAL] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss


def test(args, model, criterion, device, test_dataloader):
    torch.set_grad_enabled(False)
    model.eval()

    if args.modality == 'res':
        print("[Warning]: using residual frames as input")

    total_loss = 0.0
    correct = 0
    for i, data in enumerate(test_dataloader, 1):
        # get inputs
        sampled_clips, idxs = data
        targets = idxs.to(device)
        outputs = []
        for clips in sampled_clips:
            inputs = clips.to(device)
            # forward
            if args.modality == 'res':
                o = model(diff(inputs)) 
            else:
                o = model(inputs)
            o = torch.mean(o, dim=0)
            outputs.append(o)
        outputs = torch.stack(outputs)
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        #print('correct: {}, {}, {}'.format(correct, targets, pts))
    avg_loss = total_loss / len(test_dataloader)
    avg_acc = correct / len(test_dataloader.dataset)
    print('[TEST] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Finetune 3D CNN from VCOP pretrained weights')
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--model', type=str, default='r3d', help='c3d/r3d/r21d')
    parser.add_argument('--dataset', type=str, default='ucf101', help='ucf101/hmdb51')
    parser.add_argument('--split', type=str, default='1', help='dataset split')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--ft_lr', type=float, default=1e-3, help='finetune learning rate')
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--log', type=str, default='ckpt_cls', help='log directory')
    parser.add_argument('--ckpt', type=str, help='checkpoint path')
    parser.add_argument('--desp', type=str, help='additional description')
    parser.add_argument('--epochs', type=int, default=150, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--bs', type=int, default=8, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--pf', type=int, default=10, help='print frequency every batch')
    parser.add_argument('--seed', type=int, default=632, help='seed for initializing training.')
    parser.add_argument('--modality', default='res', type=str, help='currently support [rgb, res]')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))

    # Force the pytorch to create context on the specific device 
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ########### model ##############
    if args.dataset == 'ucf101':
        class_num = 101
    elif args.dataset == 'hmdb51':
        class_num = 51

    if args.model == 'r3d':
        model = R3DNet(layer_sizes=(1,1,1,1), with_classifier=True, num_classes=class_num).to(device)
    
    pretrained_weights = load_pretrained_weights(args.ckpt)
    model.load_state_dict(pretrained_weights, strict=False)

    if args.desp:
        exp_name = '{}_{}_cls_cl{}_{}_{}'.format(args.model, args.modality, args.cl, args.desp, time.strftime('%m%d'))
    else:
        exp_name = '{}_{}_cls_cl{}_{}'.format(args.model, args.modality, args.cl, time.strftime('%m%d'))
    print(exp_name)
    log_dir = os.path.join(args.log, exp_name)
    writer = None
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    train_transforms = transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.RandomCrop(112),
        transforms.ToTensor()
    ])

    if args.dataset == 'ucf101':
        train_dataset = UCF101Dataset('data', args.cl, args.split, True, False, train_transforms)
        val_size = 800
    else:
        train_dataset = HMDB51Dataset('data', args.cl, args.split, True, False, train_transforms)
        val_size = 400
    
    # split val for 800 videos
    train_dataset, val_dataset = random_split(train_dataset, (len(train_dataset)-val_size, val_size))
    print('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                                num_workers=args.workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False,
                                num_workers=args.workers, pin_memory=True)

    ### loss funciton, optimizer and scheduler ###
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': [param for name, param in model.named_parameters() if 'linear' not in name and 'conv5' not in name and 'conv4' not in name]},
        {'params': [param for name, param in model.named_parameters() if 'linear' in name or 'conv5' in name or 'conv4' in name], 'lr': args.ft_lr}],
        lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-5, patience=30, factor=0.1)

    prev_best_val_loss = float('inf')
    prev_best_model_path = None
    for epoch in range(args.start_epoch, args.epochs):
        time_start = time.time()
        train(args, model, criterion, optimizer, device, train_dataloader, writer, epoch)
        val_loss = validate(args, model, criterion, device, val_dataloader, writer, epoch)
        print('Epoch time: {:.2f} s.'.format(time.time() - time_start))
        scheduler.step(val_loss)
        #writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        #writer.add_scalar('train/ft_lr', optimizer.param_groups[1]['lr'], epoch)
        # save model every 20 epoches
        if epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.join(log_dir, 'model_{}.pt'.format(epoch)))
        # save model for the best val
        if val_loss < prev_best_val_loss:
            model_path = os.path.join(log_dir, 'best_model_{}.pt'.format(epoch))
            torch.save(model.state_dict(), model_path)
            prev_best_val_loss = val_loss
            if prev_best_model_path:
                os.remove(prev_best_model_path)
            prev_best_model_path = model_path

    # test, using best model
    model.load_state_dict(torch.load(prev_best_model_path))
    test_transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(112),
            transforms.ToTensor()
    ])

    if args.dataset == 'ucf101':
        test_dataset = UCF101Dataset('data/', args.cl, args.split, False, False, test_transforms, 10)
    else:
        test_dataset = HMDB51Dataset('data/', args.cl, args.split, False, False, test_transforms, 10)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    print('TEST video number: {}.'.format(len(test_dataset)))
    criterion = nn.CrossEntropyLoss()
    test(args, model, criterion, device, test_dataloader)
