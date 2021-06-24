import torch
import torch.nn as nn
import torch.optim as optim
from utils.options import args
import utils.common as utils
import os
import time
import copy
import sys
import random
from tqdm import tqdm
import numpy as np
import heapq
from data import cifar10, cifar100
from utils.common import *
from importlib import import_module

from utils.conv_type import *
from utils.indicators import l2_reg_ortho, get_ntk, ntk_differentiable
from utils.logger import prepare_logger, prepare_seed

import models
from pdb import set_trace as bp

visible_gpus_str = ','.join(str(i) for i in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpus_str
args.gpus = [i for i in range(len(args.gpus))]
checkpoint = utils.checkpoint(args)
# now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
# logger = utils.get_logger(os.path.join(args.job_dir, 'logger'+now+'.log'))
args.save_dir = args.job_dir
logger = prepare_logger(args)
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'

if args.label_smoothing is None:
    loss_func = nn.CrossEntropyLoss().cuda()
else:
    loss_func = LabelSmoothing(smoothing=args.label_smoothing)

# Data
print('==> Loading Data..')
if args.data_set == 'cifar10':
    loader = cifar10.Data(args)
elif args.data_set == 'cifar100':
    loader = cifar100.Data(args)


PID = os.getpid()


def train(model, optimizer, trainLoader, args, epoch, logger, model_dense=None):

    model.train()
    losses = utils.AverageMeter(':.4e')
    accuracy = utils.AverageMeter(':6.3f')
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10
    start_time = time.time()
    pbar = tqdm(trainLoader, position=0, leave=True)
    # for batch, (inputs, targets) in enumerate(trainLoader):
    for batch, (inputs, targets) in enumerate(pbar):
        loss = 0; output = None
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        model.zero_grad()

        # TODO
        ### supervised
        # output = model(inputs)
        # # adjust_learning_rate(optimizer, epoch, batch, print_freq, args)
        # loss = loss_func(output, targets)
        # loss.backward()

        ### Orth Reg
        # with torch.no_grad():
        #     output = model(inputs)
        # loss = l2_reg_ortho(model)
        # loss.backward()

        ### NTK cond
        # with torch.no_grad():
        #     output = model(inputs)
        # unfreeze_model_weights(model)
        # loss = get_ntk(model, inputs, targets, num_classes=10)
        # loss.backward()

        #### ntk difference
        # with torch.no_grad():
        #     output = model(inputs)
        # unfreeze_model_weights(model)
        # ntk_dense = ntk_differentiable(model_dense, inputs, train_mode=True, need_graph=False)
        # ntk = ntk_differentiable(model, inputs, train_mode=True, need_graph=True)
        # delta_ntk = 1 - torch.trace(torch.matmul(ntk, ntk_dense.T)) / torch.trace(torch.matmul(ntk, ntk.T)).sqrt() / torch.trace(torch.matmul(ntk_dense, ntk_dense.T)).sqrt()
        # # delta_ntk = nn.functional.mse_loss(ntk, ntk_dense) # TODO reweighting lambda
        # delta_ntk.backward()
        # loss += delta_ntk
        #### dense model output
        # with torch.no_grad():
        #     output_dense = model_dense(inputs)
        # unfreeze_model_weights(model)
        # output = model(inputs)
        # delta_output = nn.functional.mse_loss(output, output_dense)
        # delta_output.backward()
        # loss += delta_output

        #### ntk condition number
        unfreeze_model_weights(model)
        ntk = ntk_differentiable(model, inputs, train_mode=True, need_graph=True)
        eigenvalues, _ = torch.symeig(ntk)
        cond = eigenvalues[-1] / eigenvalues[0]
        cond.backward()
        loss += cond

        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        if output is not None:
            prec1 = utils.accuracy(output, targets)
            accuracy.update(prec1[0], inputs.size(0))

        pbar.set_description('Loss {} | Accuracy {}'.format(float(losses.avg), float(accuracy.avg)))

        # if batch % print_freq == 0 and batch != 0:
        #     current_time = time.time()
        #     cost_time = current_time - start_time
        #     logger.info(
        #         'Epoch[{}] ({}/{}):\t'
        #         'Loss {:.4f}\t'
        #         'Accuracy {:.2f}%\t\t'
        #         'Time {:.2f}s'.format(
        #             epoch, batch * args.train_batch_size, len(trainLoader.dataset),
        #             float(losses.avg), float(accuracy.avg), cost_time
        #         )
        #     )
        #     start_time = current_time
    return float(losses.avg), float(accuracy.avg)

def validate(model, testLoader, logger):
    global best_acc
    model.eval()

    losses = utils.AverageMeter(':.4e')
    accuracy = utils.AverageMeter(':6.3f')

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets)
            accuracy.update(predicted[0], inputs.size(0))

        # current_time = time.time()
        # logger.info(
        #     'Test Loss {:.4f}\tAccuracy {:.2f}%\t\tTime {:.2f}s\n'
        #     .format(float(losses.avg), float(accuracy.avg), (current_time - start_time))
        # )
    return float(losses.avg), float(accuracy.avg)

def generate_pr_cfg(model):
    cfg_len = {
        'vgg': 17,
        'resnet32': 32,
    }

    pr_cfg = []
    if args.layerwise == 'l1':
        weights = []
        for name, module in model.named_modules():
            if hasattr(module, "set_prune_rate") and name != 'fc' and name != 'classifier':
                conv_weight = module.weight.data.detach().cpu()
                weights.append(conv_weight.view(-1))
        all_weights = torch.cat(weights,0)
        preserve_num = int(all_weights.size(0) * (1 - args.prune_rate))
        preserve_weight, _ = torch.topk(torch.abs(all_weights), preserve_num)
        threshold = preserve_weight[preserve_num-1]

        #Based on the pruning threshold, the prune cfg of each layer is obtained
        for weight in weights:
            pr_cfg.append(torch.sum(torch.lt(torch.abs(weight),threshold)).item()/weight.size(0))
        pr_cfg.append(0)
    elif args.layerwise == 'uniform':
        pr_cfg = [args.prune_rate] * cfg_len[args.arch]
        pr_cfg[-1] = 0

    get_prune_rate(model, pr_cfg)

    return pr_cfg

def get_prune_rate(model, pr_cfg):
    all_params = 0
    prune_params = 0

    i = 0
    for name, module in model.named_modules():
        if hasattr(module, "set_prune_rate"):
            w = module.weight.data.detach().cpu()
            params = w.size(0) * w.size(1) * w.size(2) * w.size(3)
            all_params = all_params + params
            prune_params += int(params * pr_cfg[i])
            i += 1

    logger.info('Params Compress Rate: %.2f M/%.2f M(%.2f%%)' % ((all_params-prune_params)/1000000, all_params/1000000, 100. * prune_params / all_params))

def main():
    start_epoch = 0
    best_acc = 0.0

    prepare_seed(args.rand_seed)
    model, pr_cfg = get_model(args, logger, pretrained!='')
    if args.teacher:
        model_dense, _ = get_model(args, logger, pretrained!='', sparse=False)
    optimizer = get_optimizer(args, model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

    if args.resume == True:
        start_epoch, best_acc = resume(args, model, optimizer)

    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    # for epoch in range(start_epoch, args.num_epochs):
    epoch_bar = tqdm(range(start_epoch, args.num_epochs), position=0, leave=True)
    for epoch in epoch_bar:
        print("<< ============== JOB (PID = %d) %s ============== >>"%(PID, args.save_dir))
        train_loss, train_acc = train(model, optimizer, loader.trainLoader, args, epoch, logger, model_dense=model_dense if args.teacher else None)
        logger.writer.add_scalar("train/loss", train_loss, epoch); logger.writer.add_scalar("train/accuracy", train_acc, epoch)
        epoch_bar.set_description('Epoch {}/{} | Train {} {}'.format(epoch+1, args.num_epochs, train_loss, train_acc))
        if args.pretrained_model != '':
            # TODO only evaluate when using pretrained checkpoint
            test_loss, test_acc = validate(model, loader.testLoader, logger)
            logger.writer.add_scalar("test/loss", test_loss, epoch); logger.writer.add_scalar("test/accuracy", test_acc, epoch)
            is_best = best_acc < test_acc
            best_acc = max(best_acc, test_acc)
            epoch_bar.set_description('Epoch {}/{} | Train {} {} | Val {} {} | Best {}'.format(epoch+1, args.num_epochs, train_loss, train_acc, test_loss, test_acc, best_acc))
        scheduler.step()

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'cfg': pr_cfg,
        }
        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best accuracy: {:.3f}'.format(float(best_acc)))

def resume(args, model, optimizer):
    if os.path.exists(args.job_dir+'/checkpoint/model_last.pt'):
        print("=> Loading checkpoint ")
        checkpoint = torch.load(args.job_dir+'/checkpoint/model_last.pt')
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"=> Loaded checkpoint (epoch) {checkpoint['epoch']})")
        return start_epoch, best_acc
    else:
        print(f"=> No checkpoint found at '{args.job_dir}' '/checkpoint/")


def get_model(args, logger, pretrained=True, sparse=True):
    pr_cfg = []

    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]().to(device)
    if pretrained:
        ckpt = torch.load(args.pretrained_model, map_location=device)
        model.load_state_dict(ckpt['state_dict'], strict=False)

    # applying sparsity to the network
    pr_cfg = generate_pr_cfg(model)
    if sparse:
        set_model_prune_rate(model, pr_cfg, logger)

    if args.freeze_weights:
        freeze_model_weights(model)

    model = model.to(device)

    return model, pr_cfg

def get_optimizer(args, model):
    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and ("sparseThreshold" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    return optimizer

if __name__ == '__main__':
    main()
