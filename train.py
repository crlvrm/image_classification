import argparse
import os
import datetime
from pathlib import Path
from functools import reduce, partial
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from utils.util import yaml_load, create_path, check_cfgs, colorstr, train_one_epoch, auto_mixup, auto_prog
from utils.datasets import MyDatasets
from utils.augment import create_AugTransforms
from utils.mymodel import MyModel
from utils.logger import MyLogger
from utils.optimizer import create_Optimizer
from utils.scheduler import create_Scheduler
from utils.loss import create_Lossfn

import warnings
warnings.filterwarnings('ignore')

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
ROOT = Path(os.path.dirname(__file__))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgs', default='config/complete.yaml', help='配置文件路径')
    parser.add_argument('--resume', default='', help='中断训练权重路径')
    parser.add_argument('--sync_bn', default=False, help='是否多卡训练')
    parser.add_argument('--project', default='run', help='保存结果路径')
    parser.add_argument('--name', default='exp', help='保存结果名称')
    parser.add_argument('--local_rank', type=int, default=-1, help='多卡训练配置进程号，不用改动')
    return parser.parse_args()

def main(opt):
    assert torch.cuda.device_count() > LOCAL_RANK
    # 多卡训练初始化nccl后端
    if LOCAL_RANK != -1:
        init_process_group(backend='nccl', world_size = WORLD_SIZE, rank = LOCAL_RANK)
    # 导入配置文件中的参数
    
    cfgs = yaml_load(opt.cfgs)
    check_cfgs(cfgs)
    model_cfg = cfgs['model']
    data_cfg = cfgs['data']
    hyp_cfg = cfgs['hyp']
    # 建立结果保存路径
    save_dir = create_path(Path(opt.project) / opt.name)
    if LOCAL_RANK in {-1, 0}: save_dir.mkdir(parents=True, exist_ok=True)
    
    # log路径
    filename = Path(save_dir) / "log{}.log".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = MyLogger(filename, level=1) if LOCAL_RANK in {-1,0} else None
    if logger is not None and LOCAL_RANK in {-1, 0}:
        logger.both(cfgs)
  
    # 多卡syncBN
    if LOCAL_RANK != -1 and opt.sync_bn:
        model.model = nn.SyncBatchNorm.convert_sync_batchnorm(module=model.model)
        if LOCAL_RANK == 0:
            logger.both(f'{colorstr("yellow", "Attention")}: sync_bn is on')

    # 准备参数
    last, best = save_dir / 'last.pt', save_dir / 'best.pt'
    if LOCAL_RANK != -1:
        device = torch.device('cuda', LOCAL_RANK)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = GradScaler(enabled = (device != torch.device('cpu')))
    epochs = hyp_cfg['epochs']
    mixup, mixup_milestone = hyp_cfg['strategy']['mixup']
    warm_ep = hyp_cfg['warm_ep']
    aug_epoch = data_cfg['train']['aug_epoch']
    prog_learn = hyp_cfg['strategy']['prog_learn']
    loss_choice: str = [k for k, v in hyp_cfg['loss'].items() if v][0]
    dist_sampler = {'uniform':torch.distributions.uniform.Uniform(low=0, high=1),'beta': torch.distributions.beta.Beta(0.1, 0.1)}
  


    # 准备dataloader
    datasets = MyDatasets(cfgs['data'], LOCAL_RANK, save_dir)
    if loss_choice == 'bce':
        datasets.train_dataset.label_transforms = partial(MyDatasets.set_label_transforms, num_classes = model_cfg['num_classes'], label_smooth = hyp_cfg['label_smooth'])
        datasets.val_dataset.label_transforms = partial(MyDatasets.set_label_transforms, num_classes= model_cfg['num_classes'], label_smooth= hyp_cfg['label_smooth'])
    train_dataset, val_dataset = datasets.train_dataset, datasets.val_dataset
    data_sampler = None if LOCAL_RANK == -1 else DistributedSampler(dataset=train_dataset)
    train_dataloader = datasets.set_dataloader(dataset=train_dataset,
                                               bs=data_cfg['train']['bs'],
                                               nw=data_cfg['nw'],
                                               pin_memory=True,
                                               sampler=data_sampler,
                                               shuffle=data_sampler is None)
    if LOCAL_RANK in {-1, 0}:
        val_dataloader = datasets.set_dataloader(dataset=val_dataset,
                                                bs=data_cfg['val']['bs'],
                                                nw=data_cfg['nw'],
                                                pin_memory=False,
                                                shuffle=False)
    else: val_dataloader = None

    # 准备model
    model = MyModel(cfgs['model']).model
    model.to(device)
    
    # optimizer
    optimizer = create_Optimizer(optimizer=hyp_cfg['optimizer'],lr=hyp_cfg['lr0'],weight_decay=hyp_cfg['weight_decay'], momentum=hyp_cfg['warmup_momentum'],params=[p for p in model.parameters() if p.requires_grad])
   
    # scheduler
    scheduler = create_Scheduler(scheduler=hyp_cfg['scheduler'], optimizer=optimizer,warm_ep=hyp_cfg['warm_ep'], epochs=hyp_cfg['epochs'], lr0=hyp_cfg['lr0'],lrf_ratio=hyp_cfg['lrf_ratio'])
    
    # loss 
    if loss_choice == 'bce':
        lossfn = create_Lossfn(loss_choice)()
    else:
        lossfn = create_Lossfn(loss_choice)(label_smooth = hyp_cfg['label_smooth'])
    
    
    if prog_learn: mixup_chnodes = torch.linspace(*hyp_cfg['strategy']['mixup'][-1], 3,dtype=torch.int32).round_().tolist()[:-1]
    
    # focalloss hard
    if loss_choice == 'bce' and eval(hyp_cfg['strategy']['focal'].split()[0]):
        focal = create_Lossfn('focal')()
    else:
        focal = None
    focal_eff_epo = eval(hyp_cfg['strategy']['focal'].split()[1])
    focal_on = eval(hyp_cfg['strategy']['focal'].split()[0])

    if opt.resume:
        ckp = torch.load(opt.resume, map_location=device)
        start_epoch = ckp['epoch'] + 1
        best_avg_acc = ckp['best_avg_acc']
        model.load_state_dict(ckp['model'])
        optimizer.load_state_dict(ckp['optimizer'])
        scheduler.load_state_dict(ckp['scheduler'])
        
        if device != torch.device('cpu'):
            scaler.load_state_dict(ckp['scaler'])  
    
    if LOCAL_RANK in {-1, 0}:
        time.sleep(0.1)
        print(f"{'Epoch':>10}{'GPU_mem':>10}{'train_loss':>12}{f'val_loss':>12}{'top1_acc':>12}{'top5_acc':>12}")
    if LOCAL_RANK != -1:
            model = DDP(model, device_ids=[LOCAL_RANK])

    best_avg_acc = 0.
    start_epoch = 0
    # 训练和验证
    t0 = time.time()
    total_epoch = epochs+warm_ep
    for epoch in range(start_epoch, total_epoch):
        if epoch == 0:
            datasets.set_augment('train', sequence=None)
        
        # warm-up 训练
        if epoch == warm_ep:
            optimizer.param_groups[0]['momentum'] = hyp_cfg['momentum']
            datasets.set_augment('train', sequence=create_AugTransforms(augments=data_cfg['train']['augment'], imgsz=data_cfg['imgsz']))
        
        # 替换focal loss
        if focal_on or int(epoch-warm_ep) >= focal_eff_epo: 
            lossfn = focal 
            focal_on = False
        
        # 使用数据增强
        datasets.auto_aug_weaken(int(epoch-warm_ep), milestone=aug_epoch)

        # 渐进式训练
        if prog_learn:
            auto_prog(mixup_chnodes,dist_sampler,data_cfg, datasets, int(epoch-warm_ep))
        
        # 是否使用mixup
        is_mixup, lam = auto_mixup(mixup=mixup, epoch=int(epoch-warm_ep), milestone=mixup_milestone, device=device, dist_sampler=dist_sampler)
        
        
        avg_acc = train_one_epoch(model, train_dataloader, val_dataloader, lossfn, optimizer, scaler, device, epoch, total_epoch, logger, is_mixup, LOCAL_RANK, lam, scheduler)

        if LOCAL_RANK in {-1, 0}:
            # Best avgacc
            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc

            # Save model
            final_epoch: bool = epoch + 1 == total_epoch
            ckpt = {
                    'epoch': epoch,
                    'best_avg_acc': best_avg_acc,
                    'model': model.state_dict() if LOCAL_RANK == -1 else model.module.state_dict(),  
                    'optimizer': optimizer.state_dict(),  
                    'scheduler': scheduler.state_dict(),
                }
            if device != torch.device('cpu'):
                ckpt['scaler'] = scaler.state_dict()

            # Save last, best and delete
            torch.save(ckpt, last)
            if best_avg_acc == avg_acc:
                torch.save(ckpt, best)
            del ckpt

            # complete
            if final_epoch:
                logger.console(f'\nTraining complete ({(time.time() - t0) / 3600:.3f} hours)'
                                   f"\nResults saved to {colorstr('bold', save_dir)}"
                                   f'\nPredict:         python predict.py --weight {best} --root data/val/{colorstr("blue", "XXX_cls")} --imgsz "{data_cfg["imgsz"]}" --badcase --save_txt --choice {model_cfg["choice"]} --kwargs "{model_cfg["kwargs"]}" --class_head {loss_choice} --class_json {save_dir}/class_indices.json --num_classes {model_cfg["num_classes"]} --transforms "{data_cfg["val"]["augment"]}"'
                                   f'\nValidate:        python val.py --weight {best} --choice {model_cfg["choice"]} --kwargs "{model_cfg["kwargs"]}" --root {colorstr("blue", "data")} --imgsz "{data_cfg["imgsz"]}" --num_classes {model_cfg["num_classes"]} --transforms "{data_cfg["val"]["augment"]}"')

if __name__ == '__main__':
    opts = parse_opt()
    main(opts)