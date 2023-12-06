'''
# -*- coding: utf-8 -*-
# @Project   : afa
# @File      : eval.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2023/2/24 14:31

# @Desciption: 
'''

import argparse
import logging
import os
import sys
sys.path.append("../net_respo")
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import pourit as pourit
from utils import evaluate, imutils
from utils.AverageMeter import AverageMeter
from utils.camutils import (cam_valid, multi_scale_cam)
from net_respo.net_cam2d import CamNet
from glob import glob
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configs/liquid_ours_seen.yaml', type=str, help="config")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument("--radius", default=8, type=int, help="radius")
parser.add_argument("--crop_size", default=512, type=int, help="crop_size")
parser.add_argument('--backend', default='nccl')


def validate(writer=None, model=None, data_loader=None, cfg=None):
    gts, cams, preds_05, preds_07 = [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for n_iter, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label = data # inputs:(B,3,H,W) labels:(B,H,W) cls_label:(B,1)
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            cls, attns, feats = model(inputs)

            cls_pred = (cls>0).type(torch.int16)
            if cls_label == 1: # only calculate the F1-score of positive samples in binary classification
                _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
                avg_meter.add({"cls_score": _f1})

            ###
            _, _cams = multi_scale_cam(model, inputs, cfg.cam.scales)
            resized_cam = F.interpolate(_cams.detach(), size=labels.shape[1:], mode='bilinear', align_corners=False) #(B,1,H,W)
            valid_cam = cam_valid(resized_cam, cls_pred)

            grid_imgs, grid_cam = imutils.tensorboard_image(imgs=inputs.clone(), cam=valid_cam)
            grid_labels = imutils.tensorboard_label(labels=labels.cpu().numpy().astype(np.int16)) #(1,512,512)
            grid_preds = imutils.tensorboard_label(labels=(valid_cam>0.5).squeeze(1).cpu().numpy().astype(np.int16))

            if n_iter % 10 ==0:
                writer.add_image("test/images", grid_imgs, global_step=n_iter)
                writer.add_image("test/labels", grid_labels, global_step=n_iter)
                writer.add_image("test/preds", grid_preds, global_step=n_iter)
                writer.add_image("test/valid_cams", grid_cam, global_step=n_iter)

            os.makedirs(os.path.join(cfg.work_dir.dir, cfg.val.split), exist_ok=True)
            if n_iter % 1 == 0 :
                save_img_path = os.path.join(cfg.work_dir.dir, cfg.val.split, '{}_img.png'.format(str(n_iter).zfill(6)))
                save_imgcam_path = os.path.join(cfg.work_dir.dir, cfg.val.split, '{}_imgcam.png'.format(str(n_iter).zfill(6)))
                save_label_path = os.path.join(cfg.work_dir.dir, cfg.val.split, '{}_label.png'.format(str(n_iter).zfill(6)))
                cv2.imwrite(save_img_path, np.transpose(grid_imgs.cpu().numpy(), (1,2,0))[..., ::-1])
                cv2.imwrite(save_imgcam_path, np.transpose(grid_cam.cpu().numpy(), (1,2,0))[..., ::-1])
                cv2.imwrite(save_label_path, np.transpose(grid_labels.cpu().numpy(), (1,2,0))[..., ::-1])

            preds_05 += list((valid_cam>0.5).cpu().numpy().astype(np.int16))
            preds_07 += list((valid_cam>0.7).cpu().numpy().astype(np.int16))
            cams += list((valid_cam>0.9).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

            valid_label = torch.nonzero(cls_label[0])[:,0]
            out_cam = torch.squeeze(resized_cam)[valid_label]
            np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})

    cls_score = avg_meter.pop('cls_score')
    seg_score_05 = evaluate.scores(gts, preds_05, num_classes=cfg.dataset.num_classes)
    cam_score = evaluate.scores(gts, cams, num_classes=cfg.dataset.num_classes)
    seg_score_07 = evaluate.scores(gts, preds_07, num_classes=cfg.dataset.num_classes)

    return cls_score, cam_score, seg_score_05, seg_score_07


def eval(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    num_workers = 10
    val_dataset = pourit.PouritDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)

    print('Test = {} images'.format(len(val_loader)))

    camNet = CamNet(backbone=cfg.backbone.config,
                  stride=cfg.backbone.stride,
                  num_classes=cfg.dataset.num_classes,
                  embedding_dim=256,
                  pretrained=True,
                  pooling=args.pooling,)
    camNet.cuda()

    writer = SummaryWriter(cfg.work_dir.tb_logger_dir)

    logging.info('Validating...')
    ckpt_files = glob(os.path.join(cfg.work_dir.ckpt_dir, 'iter_*.pth'))
    ckpt_files.sort()

    for ckpt_file in ckpt_files:
        # if not '014000' in ckpt_file:
        #     continue
        logging.info('------------{}'.format(ckpt_file))
        wetr_dict = torch.load(ckpt_file)
        new_state_dict = OrderedDict()
        for k, v in wetr_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v

        camNet.load_state_dict(new_state_dict)

        '''validation'''
        cls_score, cam_score, seg_score_05, seg_score_07 = validate(writer, model=camNet, data_loader=val_loader, cfg=cfg)
        logging.info("val cls score: %.6f"%(cls_score))
        logging.info("cams score(0.9):")
        logging.info(cam_score)
        logging.info("segs score(0.5):")
        logging.info(seg_score_05)
        logging.info("segs score(0.7):")
        logging.info(seg_score_07)




if __name__ == "__main__":
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.dataset.crop_size = args.crop_size
    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.tb_logger_dir)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)

    imutils.setup_logger(filename=os.path.join(cfg.work_dir.dir, '{}.log'.format(cfg.val.split)))

    ## fix random seed
    imutils.setup_seed(1)
    eval(cfg=cfg)
