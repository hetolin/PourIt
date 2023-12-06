'''
# -*- coding: utf-8 -*-
# @Project   : afa
# @File      : pourit.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2022/8/22 09:11

# @Desciption: 
'''

import numpy as np
from torch.utils.data import Dataset
import os
import imageio
from . import transforms
import cv2

def load_img_name_list(img_name_list_path):
    img_name_list = np.loadtxt(img_name_list_path, dtype=str)
    return img_name_list

def load_cls_label_list(name_list_dir):

    return np.load(os.path.join(name_list_dir,'cls_labels_onehot.npy'), allow_pickle=True).item()

def robust_read_image(image_name):
    image = np.asarray(imageio.imread(image_name, pilmode='RGB')) #RGB format
    if len(image.shape)<3:
        image = np.stack((image, image, image), axis=-1)

    return image

def ZeroPaddingResizeCV(img, size=(512, 512), interpolation=None):
    isize = img.shape
    ih, iw, ic = isize[0], isize[1], isize[2]
    h, w = size[0], size[1]
    scale = min(w / iw, h / ih)
    new_w = int(iw * scale + 0.5)
    new_h = int(ih * scale + 0.5)

    #cv2.resize: (H,W,1)->(H,W);(H,W,3)->(H,W,3)
    img = cv2.resize(img, (new_w, new_h), interpolation)

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)

    new_img = np.zeros((h, w, ic), np.uint8)
    new_img[(h-new_h)//2:(h+new_h)//2, (w-new_w)//2:(w+new_w)//2] = img

    return new_img


class _PouritDataset(Dataset):
    def __init__(
            self,
            root_dir=None,
            name_list_dir=None,
            split='train',
            stage='train',
    ):
        super().__init__()

        self.root_dir = root_dir
        self.stage = stage
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.name_list_dir = os.path.join(name_list_dir, split + '.txt')
        self.name_list = load_img_name_list(self.name_list_dir)

        if "train" in split:
            self.img_dir = os.path.join(self.img_dir, "train")
        elif "val" in split:
            self.img_dir = os.path.join(self.img_dir, "val")
        elif "test" in split:
            self.img_dir = os.path.join(self.img_dir, "test")


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        _img_name = self.name_list[idx]

        # use RGB only
        img_name = os.path.join(self.img_dir, _img_name+'_rgb.png')
        image = robust_read_image(img_name)

        #resize to 512
        image = ZeroPaddingResizeCV(image, size=(512, 512))

        if self.stage == "train":
            label = None

        elif self.stage == "val" or self.stage == "test":
            label_dir = os.path.join(self.img_dir.replace('JPEGImages', 'Annotations'), _img_name+'_rgb.png')
            label = np.asarray(imageio.imread(label_dir))[..., None]
            if np.max(label)==255:
                label = label / 255.
            label = ZeroPaddingResizeCV(label, size=(512, 512))

        return _img_name, image, label

class PouritDataset(_PouritDataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 num_classes=21,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes
        self.color_jittor = transforms.PhotoMetricDistortion()

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image):
        img_box = None
        if self.aug:
            if self.rescale_range:
                image = transforms.random_scaling(image, scale_range=self.rescale_range)

            if self.img_fliplr:
                image = transforms.random_fliplr(image)
            #image = self.color_jittor(image)
            if self.crop_size:
                image, img_box = transforms.random_crop(image,
                                                        crop_size=self.crop_size,
                                                        mean_rgb=[0,0,0],#[123.675, 116.28, 103.53],
                                                        ignore_index=self.ignore_index)

        image = transforms.normalize_img(image)
        image = np.transpose(image, (2, 0, 1)) ## to chw

        return image, img_box


    def __getitem__(self, idx):

        img_name, image, label = super().__getitem__(idx)

        image, img_box = self.__transforms(image=image)

        cls_label = self.label_list[img_name]

        if self.stage == 'train':
            if self.aug:
                return img_name, image, cls_label, img_box
            else:
                return img_name, image, cls_label
        elif self.stage == 'val' or self.stage=='test':
            return    img_name, image, label[...,0], cls_label

# class PouritSegDataset(_PouritDataset):
#     def __init__(self,
#                  root_dir=None,
#                  name_list_dir=None,
#                  split='train',
#                  stage='train',
#                  resize_range=[512, 640],
#                  rescale_range=[0.5, 2.0],
#                  crop_size=512,
#                  img_fliplr=True,
#                  ignore_index=255,
#                  aug=False,
#                  **kwargs):
#
#         super().__init__(root_dir, name_list_dir, split, stage)
#
#         self.aug = aug
#         self.ignore_index = ignore_index
#         self.resize_range = resize_range
#         self.rescale_range = rescale_range
#         self.crop_size = crop_size
#         self.img_fliplr = img_fliplr
#         self.color_jittor = transforms.PhotoMetricDistortion()
#
#         self.label_list = load_cls_label_list(name_list_dir=name_list_dir)
#
#     def __len__(self):
#         return len(self.name_list)
#
#     def __transforms(self, image, label):
#         if self.aug:
#             '''
#             if self.resize_range:
#                 image, label = transforms.random_resize(
#                     image, label, size_range=self.resize_range)
#
#             if self.rescale_range:
#                 image, label = transforms.random_scaling(
#                     image,
#                     label,
#                     scale_range=self.rescale_range)
#             '''
#             if self.img_fliplr:
#                 image, label = transforms.random_fliplr(image, label)
#             image = self.color_jittor(image)
#             if self.crop_size:
#                 image, label = transforms.random_crop(
#                     image,
#                     label,
#                     crop_size=self.crop_size,
#                     mean_rgb=[123.675, 116.28, 103.53],
#                     ignore_index=self.ignore_index)
#
#         image = transforms.normalize_img(image)
#         ## to chw
#         image = np.transpose(image, (2, 0, 1))
#
#         return image, label
#
#     def __getitem__(self, idx):
#         img_name, image, label = super().__getitem__(idx)
#
#         image, label = self.__transforms(image=image, label=label)
#
#         cls_label = self.label_list[img_name]
#
#         return img_name, image, label[...,0], cls_label