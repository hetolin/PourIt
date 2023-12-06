'''
# -*- coding: utf-8 -*-
# @Project   : afa
# @File      : demo.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2023/2/25 10:16

# @Desciption: 
'''

# import _init_path
import argparse
import os

import numpy as np
from copy import deepcopy
import torch
from collections import OrderedDict
from omegaconf import OmegaConf

from net_respo.net_cam2d import CamNet
from ROS_tool import rosImageReceiver, rosTF

from datasets import transforms
from datasets.pourit import ZeroPaddingResizeCV
from utils.imutils import denormalize_img
from utils.camutils import (cam_valid, multi_scale_cam)

import cv2
import matplotlib.pyplot as plt
import time
import signal
from skimage import morphology

import rospy
from std_msgs.msg import Float64MultiArray
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configs/pourit_seen_ours.yaml', type=str, help="config")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")
parser.add_argument("--crop_size", default=480, type=int, help="crop_size")
parser.add_argument("--model_path", default="./logs/pourit_ours/checkpoints/iter_014000.pth", type=str, help="model_path")

def save_to_obj_pts(verts, path):
    file = open(path, 'w')
    for v in verts:
        file.write('v %f %f %f\n' % (v[0], v[1], v[2]))

    file.close()

def save_to_obj_pts_cls(verts_cls, path):
    file = open(path, 'w')
    for v in verts_cls:
        file.write('v %f %f %f %f %f %f\n' % (v[0], v[1], v[2], v[3]/255., v[4]/255., v[5]/255.))

    file.close()

class LiquidPredictor():
    def __init__(self, cfg, cam_type='kinect_azure'):
        self.cfg = cfg
        self.initialization_camnet()
        self.initialization_ros(cam_type)

        self.T_obj2cam = None
        self.T_cam2base = None

        self.pub_liquid_point = rospy.Publisher('/liquid_point', Float64MultiArray, queue_size=1)

    def initialization_camnet(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.camNet = CamNet(backbone=cfg.backbone.config,
                           stride=cfg.backbone.stride,
                           num_classes=cfg.dataset.num_classes,
                           embedding_dim=256,
                           pretrained=True,
                           pooling=args.pooling, )

        trained_state_dict = torch.load(args.model_path, map_location="cpu")
        new_state_dict = OrderedDict()
        for k, v in trained_state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v

        self.camNet.load_state_dict(state_dict=new_state_dict, strict=True)
        self.camNet.eval()
        self.camNet.to(self.device)

    def initialization_ros(self, cam_type):
        self.ros_receiver = rosImageReceiver(cam_type=cam_type, sync=False)
        self.ros_tf = rosTF()

        while not isinstance(self.ros_receiver.image_rgb, np.ndarray):
            print('wait for RGB-D images...')
            time.sleep(1.0)

    def get_pose(self):
        _, T_obj2cam, listen_success = self.ros_tf.listen_tf(self.ros_receiver.cam_frame, '/bottle')
        self.T_obj2cam = deepcopy(T_obj2cam)

        _, T_cam2base, listen_success = self.ros_tf.listen_tf(self.ros_receiver.base_frame, self.ros_receiver.cam_frame)
        self.T_cam2base = deepcopy(T_cam2base)

        return T_obj2cam, T_cam2base

    def model_liquid(self, cam_K, mask, T_obj2cam, T_cam2base):
        R_obj2cam = T_obj2cam[:3,:3]
        t_obj2cam = T_obj2cam[:3, 3].reshape(3,1)

        R_cam2base = T_cam2base[:3,:3]
        t_cam2base = T_cam2base[:3, 3].reshape(3,1)

        T_base2cam = np.linalg.inv(T_cam2base)
        R_base2cam = T_base2cam[:3,:3]
        t_base2cam = T_base2cam[:3, 3].reshape(3,1)

        # in camera frame
        v, u = mask.nonzero()
        ones = np.ones_like(v)
        uv1 = np.stack([u, v, ones], axis=0) #(3, N)
        d = np.dot(np.linalg.inv(cam_K), uv1) #(3, N)

        Rz_obj2cam = R_obj2cam[:,-1].reshape(3,1) #(3,1)
        v_gravity_base = np.array([[0],[0],[1]]) #(3,1)
        v_gravity_cam = np.dot(R_base2cam, v_gravity_base)#  + t_base2cam #(3,1) cannot add this !!!

        Rz_obj2cam_parl = np.dot(Rz_obj2cam.T, v_gravity_cam) * v_gravity_cam #(3,1)
        Rz_obj2cam_vert = Rz_obj2cam - Rz_obj2cam_parl


        kapa = np.dot(t_obj2cam.T, Rz_obj2cam_vert) / np.dot(d.T, Rz_obj2cam_vert) #(1,1)/(N,3) = (N, 3)
        points_cam = kapa.T * d

        # in base frame
        points_base = np.dot(R_cam2base, points_cam) + t_cam2base

        return points_cam, points_base, d

    @torch.no_grad()
    def inference(self, liquid_2d_only=True):
        cnt=0
        while True:
            if not liquid_2d_only:
                self.get_pose()
                if self.T_cam2base is None or self.T_obj2cam is None:
                    print('wait for T_cam2base and T_obj2cam\r', end='', flush=True)
                    continue


            img_ori = deepcopy(self.ros_receiver.image_rgb)
            img = img_ori[..., ::-1] # BGR to RGB

            img = ZeroPaddingResizeCV(img, size=(self.cfg.dataset.crop_size, self.cfg.dataset.crop_size))
            img = transforms.normalize_img(img)
            img = np.transpose(img, (2, 0, 1))
            img_tensor = torch.tensor(img).unsqueeze(0)
            img_tensor_cuda = img_tensor.to(self.device)
            img_denorm_tensor = denormalize_img(img_tensor)

            torch.cuda.synchronize()
            start_time = time.time()

            cls_pred, cam = multi_scale_cam(self.camNet.half(), inputs=img_tensor_cuda.half(), scales=[1.])
            cls_pred = (torch.sum(cls_pred)>0).type(torch.int16) #(origin, flip_origin)
            valid_cam = cam_valid(cam, cls_pred)

            torch.cuda.synchronize()
            end_time = time.time()
            print('Elapsed time = {:.0f} Hz \r'.format(1./(end_time - start_time)), end='', flush=True)

            valid_cam = valid_cam.cpu().float()
            valid_cam = valid_cam.max(dim=1)[0]
            cam_heatmap = plt.get_cmap('plasma')(valid_cam.numpy())[:,:,:,0:3]*255
            cam_heatmap = cam_heatmap[..., ::-1]
            cam_heatmap = np.ascontiguousarray(cam_heatmap)
            cam_heatmap_tensor = torch.from_numpy(cam_heatmap) #RGB to BGR
            cam_cmap_tensor = cam_heatmap_tensor.permute([0, 3, 1, 2]) #(1,3,512,512)
            cam_img = cam_cmap_tensor*0.5 + img_denorm_tensor[:, [2,1,0] ,:, :]*0.5

            cam_img_show = np.transpose(cam_img.squeeze().numpy(), (1,2,0)).astype(np.uint8)
            cam_show = np.transpose(cam_cmap_tensor.squeeze().numpy(), (1,2,0)).astype(np.uint8)

            cv2.imshow('rgb_cam', cam_img_show)
            cv2.waitKey(1)

            if not liquid_2d_only:
                mask = (valid_cam>0.7).squeeze().numpy().astype(np.uint16)
                points_cam, points_base, _ = self.model_liquid(self.ros_receiver.camera_K, mask, self.T_obj2cam, self.T_cam2base)

                if points_base.shape[1] > 0:
                    points_base_sortindex = np.argsort(points_base[-1,:]) # max2min
                    points_base_choose = points_base[:, points_base_sortindex[0]].reshape(3,1)
                else:
                    points_base_choose = [0,0,0]
                point_msg = Float64MultiArray(data=[points_base_choose[0], points_base_choose[1], points_base_choose[2]])
                self.pub_liquid_point.publish(point_msg)

                self.save_metadata(cam_img_show, cam_show, mask, cnt)

            cnt+=1

    def save_metadata(self, cam_img_show, cam_show, pred_mask, cnt, save_dir='meta_data'):
        os.makedirs(save_dir, exist_ok=True)

        img_rgbcam = deepcopy(cam_img_show)
        img_cam = deepcopy(cam_show)
        img_rgb = deepcopy(self.ros_receiver.image_rgb)
        img_depth = deepcopy(self.ros_receiver.image_depth)
        img_camK = deepcopy(self.ros_receiver.camera_K)
        img_mask = deepcopy(pred_mask)

        img_rgbcam_savefile = os.path.join(save_dir, '{}_rgbcam.png'.format(str(cnt).zfill(6)))
        img_cam_savefile = os.path.join(save_dir, '{}_cam.png'.format(str(cnt).zfill(6)))
        img_rgb_savefile = os.path.join(save_dir, '{}_rgb.png'.format(str(cnt).zfill(6)))
        img_depth_savefile = os.path.join(save_dir, '{}_depth.png'.format(str(cnt).zfill(6)))
        img_mask_savefile = os.path.join(save_dir, '{}_mask.png'.format(str(cnt).zfill(6)))
        img_camK_savefile = os.path.join(save_dir, '{}_camK.txt'.format(str(cnt).zfill(6)))
        T_obj2cam_savefile = os.path.join(save_dir, '{}_Tobj2cam.txt'.format(str(cnt).zfill(6)))
        T_cam2base_savefile = os.path.join(save_dir, '{}_Tcam2base.txt'.format(str(cnt).zfill(6)))

        cv2.imwrite(img_rgbcam_savefile, img_rgbcam)
        cv2.imwrite(img_cam_savefile, img_cam)
        cv2.imwrite(img_rgb_savefile, img_rgb)
        cv2.imwrite(img_depth_savefile, img_depth.astype(np.uint16))
        cv2.imwrite(img_mask_savefile, (img_mask*255).astype(np.uint8))
        np.savetxt(img_camK_savefile, img_camK)
        np.savetxt(T_obj2cam_savefile, self.T_obj2cam)
        np.savetxt(T_cam2base_savefile, self.T_cam2base)


    def process_metadata(self, save_dir, cnt):
        img_rgb_savefile = os.path.join(save_dir, '{}_rgb.png'.format(str(cnt).zfill(6)))
        img_depth_savefile = os.path.join(save_dir, '{}_depth.png'.format(str(cnt).zfill(6)))
        img_mask_savefile = os.path.join(save_dir, '{}_mask.png'.format(str(cnt).zfill(6)))
        img_camK_savefile = os.path.join(save_dir, '{}_camK.txt'.format(str(cnt).zfill(6)))
        T_obj2cam_savefile = os.path.join(save_dir, '{}_Tobj2cam.txt'.format(str(cnt).zfill(6)))
        T_cam2base_savefile = os.path.join(save_dir, '{}_Tcam2base.txt'.format(str(cnt).zfill(6)))

        img_rgb = cv2.imread(img_rgb_savefile)[..., ::-1]
        mask = cv2.imread(img_mask_savefile)[:,:,0]

        mask[mask==255]=1
        _skt = morphology.skeletonize(mask)
        skt = _skt.astype(np.uint8)*255

        kernel = np.ones((3, 3), dtype=np.uint8)
        skt_dilate = cv2.dilate(skt, kernel, 1) # 1:迭代次数，也就是执行几次膨胀操作

        img_depth = cv2.imread(img_depth_savefile, -1) / 1000.#[:,:,0]
        cam_K = np.loadtxt(img_camK_savefile)
        T_obj2cam = np.loadtxt(T_obj2cam_savefile)
        T_cam2base = np.loadtxt(T_cam2base_savefile)

        # points_bkg_cam = self.depth_to_pointcloud(img_depth, cam_K, img_depth<0.8)
        color_points_bkg_cam = self.depth_to_pointcloud_color(img_rgb, img_depth, cam_K, img_depth<0.8)
        color_points_bkg_base = deepcopy(color_points_bkg_cam)
        color_points_bkg_base[:, :3] = np.dot(color_points_bkg_cam[:,:3], T_cam2base[:3,:3].T) + T_cam2base[:3,3]
        points_cam, points_base, _ = self.model_liquid(cam_K, skt, T_obj2cam, T_cam2base)

        modeling_3d_path = os.path.join(save_dir.replace('src', 'dst'))
        os.makedirs(modeling_3d_path, exist_ok=True)
        save_to_obj_pts_cls(color_points_bkg_cam, os.path.join(modeling_3d_path, '{}_bkg_points.obj'.format(str(cnt).zfill(6))))
        save_to_obj_pts(points_cam.T, os.path.join(modeling_3d_path, '{}_liquid_points.obj'.format(str(cnt).zfill(6))))
        cv2.imwrite(os.path.join(modeling_3d_path, 'skt.png'), skt_dilate)

    def process_metadata_multiple(self, root_dir):
        folders = ['scene{}'.format(i+1) for i in range(3)]

        for folder in folders:
            scan_folder = os.path.join(root_dir, folder)
            print(scan_folder)
            rgb_files = glob(os.path.join(scan_folder, '*_rgb.png'))

            # get index
            for rgb_file in tqdm(rgb_files):
                cnt = rgb_file.split('/')[-1].split('_')[0]
                # generate point cloud
                self.process_metadata(scan_folder, cnt)

    @staticmethod
    def depth_to_pointcloud(depth, K, pred_mask):
        depth[depth==np.nan]=0
        depth = depth * pred_mask
        vs, us = depth.nonzero()
        zs = depth[vs, us]

        xs = (us - K[0, 2]) * zs / K[0, 0]
        ys = (vs - K[1, 2]) * zs / K[1, 1]
        pts = np.stack([xs, ys, zs], axis=1)

        pts[:, 1] = pts[:, 1]
        pts[:, 2] = pts[:, 2]

        return pts

    @staticmethod
    def depth_to_pointcloud_color(rgb, depth, K, pred_mask):
        depth[depth==np.nan]=0
        depth = depth * pred_mask
        vs, us = depth.nonzero()
        zs = depth[vs, us]

        xs = (us - K[0, 2]) * zs / K[0, 0]
        ys = (vs - K[1, 2]) * zs / K[1, 1]
        pts = np.stack([xs, ys, zs], axis=1)

        pts[:, 1] = pts[:, 1]
        pts[:, 2] = pts[:, 2]

        cls = rgb[vs, us, :]

        pts_cls = np.concatenate([pts, cls], axis=1) #(N,6)

        return pts_cls


if __name__ == "__main__":
    signal.signal(signal.SIGINT, quit)
    signal.signal(signal.SIGTERM, quit)

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.dataset.crop_size = args.crop_size
    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir

    predictor = LiquidPredictor(cfg, cam_type='kinect_azure')

    # online 2d liquid prediction
    predictor.inference(liquid_2d_only=True)

    # online 3d liquid prediction
    # predictor.inference()

    # offline 3d liquid prediction
    # predictor.process_metadata_multiple('./examples/src')
