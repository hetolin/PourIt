'''
# -*- coding: utf-8 -*-
# @Project   : afa
# @File      : train.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2023/2/16 11:05
'''

import argparse
import datetime
import logging
import os
import sys
sys.path.append(".")
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from datasets import pourit as pourit
from utils import imutils
from utils.AverageMeter import AverageMeter
from utils.camutils import (cam_valid, multi_scale_cam)
from utils.optimizer import PolyWarmupAdamW
from utils.losses import get_classification_loss, get_contrast_loss
from net_respo.net_cam2d import CamNet
from scripts.eval import validate

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/test.yaml',
                    type=str,
                    help="config")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument("--radius", default=8, type=int, help="radius")
parser.add_argument("--crop_size", default=512, type=int, help="crop_size")
parser.add_argument('--backend', default='nccl')

def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

def train(cfg):

    num_workers = 10

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend,)

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = pourit.PouritDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    val_dataset = pourit.PouritDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    train_sampler = DistributedSampler(train_dataset,shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,
                              # shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True,
                              sampler=train_sampler,
                              prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)

    print('Train = {} images, Test = {} images'.format(train_loader.__len__(), len(val_loader)))
    device = torch.device(args.local_rank)

    camNet = CamNet(backbone=cfg.backbone.config,
                  stride=cfg.backbone.stride,
                  num_classes=cfg.dataset.num_classes,
                  embedding_dim=256,
                  pretrained=True,
                  pooling=args.pooling, )
    logging.info('\nNetwork config: \n%s'%(camNet))
    param_groups = camNet.get_param_groups()
    camNet.to(device)

    writer = SummaryWriter(cfg.work_dir.tb_logger_dir)

    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0, ## freeze norm layers
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr = cfg.optimizer.learning_rate,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = cfg.scheduler.warmup_iter,
        max_iter = cfg.train.max_iters,
        warmup_ratio = cfg.scheduler.warmup_ratio,
        power = cfg.scheduler.power
    )
    logging.info('\nOptimizer: \n%s' % optimizer)
    camNet = DistributedDataParallel(camNet, device_ids=[args.local_rank], find_unused_parameters=True)

    train_sampler.set_epoch(np.random.randint(cfg.train.max_iters))
    train_loader_iter = iter(train_loader)

    avg_meter = AverageMeter()

    camNet.train()
    for n_iter in range(cfg.train.max_iters):

        try:
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)
        except:
            train_sampler.set_epoch(np.random.randint(cfg.train.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        cls_labels = cls_labels.to(device, non_blocking=True)

        cls, attns, feats = camNet(inputs)


        _, cams = multi_scale_cam(camNet, inputs=inputs, scales=cfg.cam.scales)
        valid_cam = cam_valid(cams.detach(), cls_label=cls_labels)
        valid_cam_resized = F.interpolate(valid_cam, size=(feats.shape[2], feats.shape[3]), mode='bilinear', align_corners=False) #(B, 1, h, w)

        cls_loss = get_classification_loss(cls, cls_labels)
        if n_iter <= cfg.train.cam_iters:
            neg_loss = torch.Tensor([0.]).cuda()
            pos_loss = torch.Tensor([0.]).cuda()

            loss = 1.0 * cls_loss #+ 0.0 * neg_loss + 0.0 * pos_loss
        else:
            pos_ids = torch.nonzero(valid_cam_resized>=0.7)
            neg_ids = torch.nonzero(valid_cam_resized<0.7)
            pos_feats = feats[pos_ids[:, 0], :, pos_ids[:, 2], pos_ids[:, 3]] #(N,C)
            neg_feats = feats[neg_ids[:, 0], :, neg_ids[:, 2], neg_ids[:, 3]]

            #calculate pos_loss and neg_loss
            pos_loss, neg_loss = get_contrast_loss(pos_feats, neg_feats, cls_labels)

            loss = 1.0 * cls_loss + 1.0 * neg_loss + 1.0 * pos_loss #+ 0.0 * torch.sum(attn_pred)

        avg_meter.add({'cls_loss': cls_loss.item(), 'neg_loss': neg_loss.item(), 'pos_loss': pos_loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (n_iter+1) % cfg.train.log_iters == 0:
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']
            grid_imgs, grid_cam = imutils.tensorboard_image(imgs=inputs.clone(), cam=valid_cam)
            grid_fmap4 = imutils.tensorboard_fmap(feats)

            if args.local_rank==0:
                logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, neg_loss %.4f, pos_loss %.4f"%(n_iter+1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('neg_loss'), avg_meter.pop('pos_loss')))

                writer.add_image("train/images", grid_imgs, global_step=n_iter)
                writer.add_image("cam/valid_cams", grid_cam, global_step=n_iter)
                writer.add_image("feature_map/block1", grid_fmap4.clone(), global_step=n_iter)


        if (n_iter+1) % cfg.train.eval_iters == 0:
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "iter_{}.pth".format(str(n_iter+1).zfill(6)))
            if args.local_rank==0:
                logging.info('Validating...')
                torch.save(camNet.state_dict(), ckpt_name)

            '''validation'''
            # cls_score, seg_score, cam_score, seg_highconf_score = validate(writer, model=camNet, data_loader=val_loader, cfg=cfg)
            # if args.local_rank==0:
            #     logging.info("val cls score: %.6f"%(cls_score))
            #     logging.info("cams score:")
            #     logging.info(cam_score)
            #     logging.info("segs score(0.5):")
            #     logging.info(seg_score)
            #     logging.info("segs score(0.7):")
            #     logging.info(seg_highconf_score)

    return True


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.dataset.crop_size = args.crop_size
    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.tb_logger_dir)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)

    if args.local_rank == 0:
        imutils.setup_logger(filename=os.path.join(cfg.work_dir.dir, 'train.log'))
        logging.info('\nargs: %s' % args)
        logging.info('\nconfigs: %s' % cfg)

    ## fix random seed
    imutils.setup_seed(1)
    train(cfg=cfg)
