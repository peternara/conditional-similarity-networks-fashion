from __future__ import print_function
import argparse
import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from triplet_image_loader import TripletImageLoader
from tripletnet import CS_Tripletnet
from visdom import Visdom
import numpy as np

import Resnet_18
from csn import ConditionalSimNet


import tensorboardcolab as tb
from torch.utils.tensorboard import SummaryWriter

from google.colab import drive
drive.mount('/content/gdrive')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64) 256 ')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                    help='learning rate (default: 5e-5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Conditional_Similarity_Network', type=str,
                    help='name of experiment')
parser.add_argument('--embed_loss', type=float, default=5e-3, metavar='M',
                    help='parameter for loss for embedding norm')
parser.add_argument('--mask_loss', type=float, default=5e-4, metavar='M',
                    help='parameter for loss for mask norm')
parser.add_argument('--num_traintriplets', type=int, default=100000, metavar='N',
                    help='how many unique training triplets (default: 100000)')
parser.add_argument('--dim_embed', type=int, default=64, metavar='N',
                    help='how many dimensions in embedding (default: 64)')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--learned', dest='learned', action='store_true',
                    help='To learn masks from random initialization')
parser.add_argument('--prein', dest='prein', action='store_true',
                    help='To initialize masks to be disjoint')
parser.add_argument('--visdom', dest='visdom', action='store_true',
                    help='Use visdom to track and plot')
parser.add_argument('--conditions', nargs='*', type=int,
                    help='Set of similarity notions')
parser.set_defaults(test=False)
parser.set_defaults(learned=True) # False
parser.set_defaults(prein=False)
parser.set_defaults(visdom=False)

best_acc = 0

step = 0

def main():
    global args, best_acc
    args      = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    #torch.manual_seed(args.seed)    
    if args.cuda:
        #torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda:0")


    if args.visdom:
        global plotter 
        plotter = VisdomLinePlotter(env_name=args.name)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    global conditions
    if args.conditions is not None:
        conditions = args.conditions
    else:
        conditions = [0,1,2,3]
    
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    train_loader = torch.utils.data.DataLoader(
        TripletImageLoader('data', 'ut-zap50k-images', 'filenames.json', 
            conditions, 'train', n_triplets=args.num_traintriplets,
                        transform=transforms.Compose([
                            transforms.Resize(112),
                            transforms.CenterCrop(112),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        TripletImageLoader('data', 'ut-zap50k-images', 'filenames.json', 
            conditions, 'test', n_triplets=160000,
                        transform=transforms.Compose([
                            transforms.Resize(112),
                            transforms.CenterCrop(112),
                            transforms.ToTensor(),
                            normalize,
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(
        TripletImageLoader('data', 'ut-zap50k-images', 'filenames.json', 
            conditions, 'val', n_triplets=80000,
                        transform=transforms.Compose([
                            transforms.Resize(112),
                            transforms.CenterCrop(112),
                            transforms.ToTensor(),
                            normalize,
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    model     = Resnet_18.resnet18(pretrained=True, embedding_size=args.dim_embed)
    model.to(device)

    csn_model = ConditionalSimNet(model, n_conditions=len(conditions), 
        embedding_size=args.dim_embed, learnedmask=args.learned, prein=args.prein)

    global mask_var
    mask_var = csn_model.masks.weight
    tnet     = CS_Tripletnet(csn_model)
    if args.cuda:
        tnet.cuda()
        
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint       = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1       = checkpoint['best_prec1']
            tnet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    criterion  = torch.nn.MarginRankingLoss(margin = args.margin)
    parameters = filter(lambda p: p.requires_grad, tnet.parameters())
    optimizer  = optim.Adam(parameters, lr=args.lr)

    print('=================================================================')
    print('learned      : ', args.learned)
    print('batch-size   : ', args.batch_size)
    print('conditions   : ', args.conditions)
    print('margin       : ', args.margin)
    print('resume       : ', args.resume)
    print('train_loader : ', len(train_loader))
    print('test_loader  : ', len(test_loader))
    print('val_loader   : ', len(val_loader))

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    if args.test:
        test_acc = test(test_loader, tnet, criterion, 1)
        sys.exit() 
    
    #tb = tb.TensorBoardColab()
    tb = SummaryWriter() 
    
    for epoch in range(args.start_epoch, args.epochs + 1):
        # update learning rate
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, tnet, criterion, optimizer, epoch, tb)

        # evaluate on validation set
        acc = test(val_loader, tnet, criterion, epoch)

        # remember best acc and save checkpoint
        is_best  = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': tnet.state_dict(),
            'best_prec1': best_acc,
        }, is_best)

    tb.close()   

def train(train_loader, tnet, criterion, optimizer, epoch, tb):
    losses     = AverageMeter()
    accs       = AverageMeter()
    emb_norms  = AverageMeter()
    mask_norms = AverageMeter()

    global step
    batch_iterator = iter(train_loader)

    # switch to train mode
    tnet.train()
    #for batch_idx, (data1, data2, data3, c) in enumerate(train_loader):
    for batch_idx in range(len(train_loader.dataset)):      
        try:
            data = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(train_loader)
            data           = next(batch_iterator)
        except Exception as e:
            print('Loading data exception:', e)  
            continue

        data1, data2, data3, c = data    

        if args.cuda:
            data1, data2, data3, c = data1.cuda(), data2.cuda(), data3.cuda(), c.cuda()
        data1, data2, data3, c = Variable(data1), Variable(data2), Variable(data3), Variable(c)
        
        # torch.Size([128, 3, 112, 112]) torch.Size([128, 3, 112, 112]) torch.Size([128, 3, 112, 112]) torch.Size([128])
        # print(data1.shape, data2.shape, data3.shape, c.shape)

        # compute output
        dista, distb, mask_norm, embed_norm, mask_embed_norm = tnet(data1, data2, data3, c)

        # 1 means, dista should be larger than distb
        #print('dista : ', dista.size()) # [32] > batch size
        #print(dista) # [0.4902, 0.3963, 0.4873, 0.5724, 0.4054, 0.3121, 0.3828, 0.4504, 0.5122,
                      #  0.5277, 0.4292, 0.4265, 0.3459, 0.2941, 0.5692, 0.5078, 0.3934, 0.4100,
                      #  0.5169, 0.3513, 0.4341, 0.3800, 0.4838, 0.4658, 0.3866, 0.4130, 0.4975,
                      #  0.4475, 0.4680, 0.4000, 0.4422, 0.4319]       
        #print('distb : ', distb.size()) # [32] > batch size
        #print(distb) # [0.4969, 0.2613, 0.3864, 0.2541, 0.3750, 0.3438, 0.4856, 0.3948, 0.4113,
                      #  0.3821, 0.4903, 0.3686, 0.5303, 0.4397, 0.4437, 0.3278, 0.3889, 0.3586,
                      #  0.3326, 0.3111, 0.4075, 0.4273, 0.4425, 0.3362, 0.2934, 0.3503, 0.4526,
                      #  0.5191, 0.3576, 0.4157, 0.2716, 0.5081]

        # MarginRankingLoss 
        # dista > distb 이니 모두 1인 케이스다
        #     만약 ranking이 적다면 -1로 표시해주는 ,,,
        # If :math:`y = 1` then it assumed the first input should be ranked higher, 
        #           (have a larger value) than the second input, and vice-versa for :math:`y = -1`.
        # \text{loss}(x, y) = \max(0, -y * (x1 - x2) + \text{margin})
        target = torch.FloatTensor(dista.size()).fill_(1)
        #print('target : ', target) # tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                   #              1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]) 
        #print('\t', target.shape) # [32] > batch size        

        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        
        loss_triplet = criterion(dista, distb, target)
        loss_embedd  = embed_norm / np.sqrt(data1.size(0))
        loss_mask    = mask_norm / data1.size(0)
        loss         = loss_triplet + args.embed_loss * loss_embedd + args.mask_loss * loss_mask

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        losses.update(loss_triplet.data.item(), data1.size(0))
        accs.update(acc, data1.size(0))
        emb_norms.update(loss_embedd.data.item())
        mask_norms.update(loss_mask.data.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                losses.val, losses.avg, 
                100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))                
            #tb.add_scalar("accs.val", accs.val, step) 
            tb.add_scalar("loss", losses.avg, step)  
            tb.add_scalar("accs.avg", accs.avg, step)
            step+=1       

    # log avg values to visdom
    if args.visdom:
        plotter.plot('acc', 'train', epoch, accs.avg)
        plotter.plot('loss', 'train', epoch, losses.avg)
        plotter.plot('emb_norms', 'train', epoch, emb_norms.avg)
        plotter.plot('mask_norms', 'train', epoch, mask_norms.avg)
        if epoch % 10 == 0:
            plotter.plot_mask(torch.nn.functional.relu(mask_var).data.cpu().numpy().T, epoch)

def test(test_loader, tnet, criterion, epoch, tb):
    losses  = AverageMeter()
    accs    = AverageMeter()
    accs_cs = {}
    for condition in conditions:
        accs_cs[condition] = AverageMeter()

    batch_iterator = iter(test_loader)    

    # switch to evaluation mode
    tnet.eval()
    #for batch_idx, (data1, data2, data3, c) in enumerate(test_loader):

    for batch_idx in range(len(test_loader.dataset)):
        try:
            data = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(test_loader)
            data           = next(batch_iterator)
        except Exception as e:
            print('Loading data exception:', e)  
            continue

        data1, data2, data3, c = data   

        if args.cuda:
            data1, data2, data3, c = data1.cuda(), data2.cuda(), data3.cuda(), c.cuda()
        data1, data2, data3, c = Variable(data1), Variable(data2), Variable(data3), Variable(c)
        c_test = c

        # compute output
        dista, distb, _, _, _ = tnet(data1, data2, data3, c)
        target                = torch.FloatTensor(dista.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target    = Variable(target)
        test_loss =  criterion(dista, distb, target).data.item()

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        accs.update(acc, data1.size(0))
        for condition in conditions:
            accs_cs[condition].update(accuracy_id(dista, distb, c_test, condition), data1.size(0))
        losses.update(test_loss, data1.size(0))      

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(losses.avg, 100. * accs.avg))
    tb.add_scalar("test_acc", losses.avg, epoch) 

    if args.visdom:
        for condition in conditions:
            plotter.plot('accs', 'acc_{}'.format(condition), epoch, accs_cs[condition].avg)
        plotter.plot(args.name, args.name, epoch, accs.avg, env='overview')
        plotter.plot('acc', 'test', epoch, accs.avg)
        plotter.plot('loss', 'test', epoch, losses.avg)

    return accs.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y, env=None):
        if env is not None:
            print_env = env
        else:
            print_env = self.env
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=print_env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=print_env, win=self.plots[var_name], name=split_name)
    def plot_mask(self, masks, epoch):
        self.viz.bar(
            X=masks,
            env=self.env,
            opts=dict(
                stacked=True,
                title=epoch,
            )
        )

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * ((1 - 0.015) ** epoch)
    if args.visdom:
        plotter.plot('lr', 'learning rate', epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum()*1.0/dista.size()[0]

def accuracy_id(dista, distb, c, c_id):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return ((pred > 0)*(c.cpu().data == c_id)).sum()*1.0/(c.cpu().data == c_id).sum()

if __name__ == '__main__':
    main()    
