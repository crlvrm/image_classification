import yaml
from pathlib import Path
import os
from functools import reduce, partial
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
from typing import Callable, Optional
import torch.nn as nn
from .augment import Mixup, CenterCropAndResize
from torchvision.transforms import Compose, CenterCrop, Resize


def yaml_load(file=''):
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)

def create_path(path, exist_ok=False, sep='', mkdir=False):
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)
    
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)

    return path

def check_cfgs(cfgs):
    model_cfg = cfgs['model']
    data_cfg = cfgs['data']
    hyp_cfg = cfgs['hyp']
    # model
    assert model_cfg['choice'].split('-')[0] in {'torchvision', 'custom'}, 'if from torchvision, torchvision-ModelName; if from your own, custom-ModelName'
    if model_cfg['kwargs'] and model_cfg['pretrained']:
        for k in model_cfg['kwargs'].keys():
            if k not in {'dropout'}: raise KeyError('set kwargs except dropout, pretrained must be False')
    assert (model_cfg['pretrained'] and ('normalize' in data_cfg['train']['augment'].split()) and ('normalize' in data_cfg['val']['augment'].split())) or \
           (not model_cfg['pretrained']) and ('normalize' not in data_cfg['train']['augment'].split()) and ('normalize' not in data_cfg['val']['augment'].split()),\
           'if not pretrained, normalize is not necessary, or normalize is necessary'
    # loss
    assert reduce(lambda x, y: int(x) + int(y), list(hyp_cfg['loss'].values())) == 1, 'ce or bce'
    # optimizer
    assert hyp_cfg['optimizer'] in {'sgd', 'adam'}, 'optimizer choose sgd or adam'
    # scheduler
    assert hyp_cfg['scheduler'] in {'linear', 'cosine', 'linear_with_warm', 'cosine_with_warm'}, 'scheduler support linear cosine linear_with_warm and cosine_with_warm'
    assert hyp_cfg['warm_ep'] >= 0 and isinstance(hyp_cfg['warm_ep'], int) and hyp_cfg['warm_ep'] < hyp_cfg['epochs'], 'warm_ep not be negtive, and should smaller than epochs'
    if hyp_cfg['warm_ep'] == 0: assert hyp_cfg['scheduler'] in {'linear', 'cosine'}, 'no warm, linear or cosine supported'
    if hyp_cfg['warm_ep'] > 0: assert hyp_cfg['scheduler'] in {'linear_with_warm', 'cosine_with_warm'}, 'with warm, linear_with_warm or cosine_with_warm supported'
    # strategy
    # focalloss
    if eval(hyp_cfg['strategy']['focal'].split()[0]): assert hyp_cfg['loss']['bce'], 'focalloss only support bceloss'
    # mixup
    mixup, mixup_milestone = map(eval, hyp_cfg['strategy']['mixup'].split())
    assert mixup >= 0 and mixup <= 1 and isinstance(mixup_milestone, list), 'mixup_ratio[0,1], mixup_milestone be list'
    mix0, mix1 = mixup_milestone
    assert isinstance(mix0, int) and isinstance(mix1, int) and mix0 < mix1, 'mixup must List[int], start < end'
    hyp_cfg['strategy']['mixup'] = [mixup, mixup_milestone]
    # progressive learning
    if hyp_cfg['strategy']['prog_learn']: assert mixup > 0 and data_cfg['train']['aug_epoch'] >= mix1, 'if progressive learning, make sure mixup > 0, and aug_epoch >= mix_end'
    # augment
    augs = ['center_crop', 'resize']
    train_augs = data_cfg['train']['augment'].split()
    val_augs = data_cfg['val']['augment'].split()
    assert not (augs[0] in train_augs and augs[1] in train_augs), 'if need centercrop and resize, please centercrop offline, not support use two'
    for a in augs:
        if a in train_augs:
            assert a in val_augs, 'augment about image size should be same in train and val'
    n_val_augs = len(val_augs)
    assert train_augs[-n_val_augs:] == val_augs, 'augment in val should be same with end-part of train'

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def auto_mixup(mixup: float, epoch:int, milestone: list, device, dist_sampler):
      
    if mixup == 0 or epoch >= milestone[1] or epoch < milestone[0]: return (False, None) # is_mixup, lam
    else:
        mix_prob = dist_sampler['uniform'].sample()
        is_mixup: bool = mix_prob < mixup
        lam = dist_sampler['beta'].sample().to(device)
        return is_mixup.item() if isinstance(is_mixup, torch.Tensor) else is_mixup, lam
    
def auto_prog(mixup_chnodes,dist_sampler,data_cfg, datasets, epoch):
        chnodes = mixup_chnodes
        # mixup, divide mixup_milestone into 2 parts in default, alpha from 0.1 to 0.2
        if epoch in chnodes:
            alpha = (mixup_chnodes.index(epoch) + 1) * 0.1
            dist_sampler['beta'] = torch.distributions.beta.Beta(alpha, alpha)
        # image resize, based on mixup_milestone
        min_imgsz = min(data_cfg['imgsz']) if isinstance(data_cfg['imgsz'][0], int) else min(data_cfg['imgsz'][-1])
        imgsz_milestone = torch.linspace(int(min_imgsz * 0.5), int(min_imgsz), 3, dtype=torch.int32).tolist()
        sequence = []
        if epoch == 0: size = imgsz_milestone[0]
        elif epoch == chnodes[0]: size = imgsz_milestone[1]
        elif epoch == chnodes[1]: size = imgsz_milestone[2]
        else: return
        train_augs = datasets.train_dataset.transforms.transforms
        for i, m in enumerate(train_augs):
            if isinstance(m, CenterCrop):
                if i+1 < len(train_augs) and not isinstance(train_augs[i+1], Resize):
                    sequence.extend([m, Resize(size)])
                else:
                    sequence.append(m)
            elif isinstance(m, Resize):sequence.append(Resize(size))
            elif isinstance(m, CenterCropAndResize):
                m[-1] = Resize(size)
                sequence.append(m)
            else: sequence.append(m)
        datasets.set_augment('train', sequence=Compose(sequence))

def print_imgsz(images: torch.Tensor):
    h, w = images.shape[-2:]
    return [h,w]

def update(model, loss, scaler, optimizer):
    # backward
    scaler.scale(loss).backward()

    # optimize
    scaler.unscale_(optimizer)  # unscale gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad()    


def train_one_epoch(model, train_dataloader, val_dataloader, criterion, optimizer,
                    scaler, device: torch.device, epoch: int,
                    epochs: int, logger, is_mixup: bool, rank: int,
                    lam, schduler):
    # train mode
    model.train()

    cuda: bool = device != torch.device('cpu')

    if rank != -1:
        train_dataloader.sampler.set_epoch(epoch)
    pbar = enumerate(train_dataloader)
    if rank in {-1, 0}:
        pbar = tqdm(enumerate(train_dataloader),
                    total=len(train_dataloader),
                    bar_format='{l_bar}{bar:10}{r_bar}')

    tloss, avg_acc = 0., 0.,

    for i, (images, labels) in pbar:  # progress bar
        images, labels = images.to(device, non_blocking=True), labels.to(device)

        with torch.cuda.amp.autocast(enabled=cuda):
            # mixup
            if is_mixup:
                images, targets_a, targets_b = Mixup.mixup_data(images, labels, device, lam)
                loss = Mixup.mixup_criterion(criterion, model(images), targets_a, targets_b, lam)
            else:
                loss = criterion(model(images), labels)
        # scale + backward + grad_clip + step + zero_grad
        update(model, loss, scaler, optimizer)

        if rank in {-1, 0}:
            tloss = (tloss * i + loss.item()) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if cuda else 0)  # (GB)
            pbar.desc = f"{f'{epoch + 1}/{epochs}':>10}{mem:>10}{tloss:>12.3g}" + ' ' * 36
            pbar.postfix = f'lr:{optimizer.param_groups[0]["lr"]:.5f}, imgsz:{print_imgsz(images)}'

            if i == len(pbar) - 1:  # last batch
                logger.log(f'epoch:{epoch + 1:d}  t_loss:{tloss:4f}  lr:{optimizer.param_groups[0]["lr"]:.5f}')
                logger.log(f'{"name":<8}{"nums":>8}{"top1":>10}{"top5":>10}')

                # val
                top1, top5, v_loss = val(model, val_dataloader, device, pbar, True, criterion, logger)
                logger.log(f'v_loss:{v_loss:4f}  mtop1:{top1:.3g}  mtop5:{top5:.3g}\n')

                avg_acc = top1  # define avg_acc as top1 accuracy

    schduler.step()  # step epoch-wise

    return avg_acc


def val(model: nn.Module, dataloader, device: torch.device, pbar, is_training: bool = False, lossfn: Optional[Callable] = None, logger = None):

    # eval mode
    model.eval()

    n = len(dataloader)  # number of batches
    action = 'validating'
    desc = f'{pbar.desc[:-36]}{action:>36}' if pbar else f'{action}'
    bar = tqdm(dataloader, desc, n, not is_training, bar_format='{l_bar}{bar:10}{r_bar}', position=0)
    pred, targets, loss = [], [], 0
    with torch.no_grad(): # w/o this op, computation graph will be save
        with autocast(enabled=(device != torch.device('cpu'))):
            for images, labels in bar:
                images, labels = images.to(device, non_blocking = True), labels.to(device)
                y = model(images)
                pred.append(y.argsort(1, descending=True)[:, :5])
                targets.append(labels)
                if lossfn:
                    loss += lossfn(y, labels)

    loss /= n
    pred, targets = torch.cat(pred), torch.cat(targets)
    if targets.dim() > 1: targets = torch.argmax(targets, dim=-1) # bce label, only used in training in order to compute loss
    correct = (targets[:, None] == pred).float()
    acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
    top1, top5 = acc.mean(0).tolist()
    if pbar:
        pbar.desc = f'{pbar.desc[:-36]}{loss:>12.3g}{top1:>12.3g}{top5:>12.3g}'

    if not is_training: logger.console(f'{"name":<8}{"nums":>8}{"top1":>10}{"top5":>10}{"Predict":>10}{"F1_score":>10}')
    for i, c in enumerate(dataloader.dataset.class_indices):
        pre_i = (pred[:,0]==i).float()
        acc_i = acc[targets == i]
        top1i, top5i = acc_i.mean(0).tolist()
        P_i = acc_i[:,0].sum()/pre_i.sum()
        f1_i = 2*P_i*top1i/(P_i+top1i)
        if not is_training: logger.console(f'{c:<8}{acc_i.shape[0]:>8}{top1i:>10.3f}{top5i:>10.3f}{P_i:>10.3f}{f1_i:>10.3f}')
        else: logger.log(f'{c:<8}{acc_i.shape[0]:>8}{top1i:>10.3f}{top5i:>10.3f}{P_i:>10.3f}{f1_i:>10.3f}')

    if not is_training: logger.console(f'mtop1:{top1:.3f}, mtop5:{top5:.3f}')

    if lossfn: return top1, top5, loss
    else: return top1, top5

