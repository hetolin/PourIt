'''
# -*- coding: utf-8 -*-
# @Project   : afa
# @File      : process_origin_data.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2022/8/22 10:19

# @Desciption: 
'''

import numpy as np
import os
import cv2
from glob import glob
from tqdm import tqdm
import random

class Processor():
    def __init__(self, ori_data_root, pro_data_root, npy_root, have_labelImg=False):
        self.ori_data_root = ori_data_root
        self.pro_data_root = pro_data_root
        self.npy_root = npy_root
        self.have_labelImg = have_labelImg


        # glob all image [water + water_no]
        self.water_no_list=[]
        self.water_list=[]
        for sub_ori_data_root in self.ori_data_root:
            water_no_path = os.path.join(sub_ori_data_root, 'water_no', '*_rgb.png')
            water_path = os.path.join(sub_ori_data_root, 'water', '*_rgb.png')
            if len(glob(water_path))==0:
                water_no_path = os.path.join(sub_ori_data_root, 'water_no', '*.png')
                water_path = os.path.join(sub_ori_data_root, 'water', '*.png')

            self.water_no_list.extend(glob(water_no_path))
            self.water_list.extend(glob(water_path))

        self.water_list.sort()
        self.water_no_list.sort()
        # self.water_label_list = [_path.replace('rgb', 'mask') for _path in self.water_list]
        # self.water_no_label_list = [_path.replace('rgb', 'mask') for _path in self.water_no_list]

        self.onehot_dict = {}

        self.class_id_dict = {'water_no':0, 'water':1}

        self.cnt = 0
        self.saveTxt_list = []

    def generate_onehot_array(self, label):
        class_num = len(self.class_id_dict)
        assert class_num >=2, 'class num should be >= 2'
        if class_num==2:
            return np.array([label]).astype(np.float32)
        else:
            return np.eye(class_num)[label].astype(np.float32)

    def remap_img(self, img_path_list, label, save_dir, save_anno_dir):


        for img_path in tqdm(img_path_list):
            # save rgb
            img = cv2.imread(img_path)
            if 'liquid' in img_path:
                img = img[90:90+300, 170:170+300, :]
            save_file_name_prefix = str(self.cnt).zfill(6)
            save_file_name = '{}_rgb.png'.format(save_file_name_prefix)
            save_path = os.path.join(save_dir, save_file_name)
            cv2.imwrite(save_path, img)

            # save mask
            if self.have_labelImg:
                mask = cv2.imread(img_path.replace('rgb', 'mask'))[...,0]
                if 'liquid' in img_path:
                    mask = mask[90:90+300, 170:170+300]

                # mask = (mask/255.).astype(np.uint8)
                save_file_name = '{}_rgb.png'.format(save_file_name_prefix)
                save_path = os.path.join(save_anno_dir, save_file_name)
                cv2.imwrite(save_path, mask)

            self.saveTxt_list.append(save_file_name_prefix)
            self.onehot_dict[save_file_name_prefix] = self.generate_onehot_array(label)
            self.cnt += 1

    def _run(self, split_list, split):
        water_list, water_no_list = split_list
        print('generate {} data'.format(split))

        # create ${split} save root
        save_dir = os.path.join(self.pro_data_root, 'JPEGImages', split)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_anno_dir = os.path.join(self.pro_data_root, 'Annotations', split)
        if not os.path.exists(save_anno_dir):
            os.makedirs(save_anno_dir)

        start_cnt = self.cnt
        print('1. remapping data [water]')
        self.remap_img(water_list, self.class_id_dict['water'], save_dir, save_anno_dir)
        print('2. remapping data [water_no]')
        self.remap_img(water_no_list, self.class_id_dict['water_no'], save_dir, save_anno_dir)


        # save ${split}.txt
        os.makedirs(self.npy_root, exist_ok=True)
        with open(os.path.join(self.npy_root, '{}.txt').format(split), 'w') as file:
            for line in self.saveTxt_list[start_cnt:]:
                file.write(line+'\n')
        print('3. save {}.txt done!'.format(split))
        print('\n')

    def run(self, ratio, shuffle=True):
        self.water_train_list, self.water_test_list = data_split(self.water_list, ratio, shuffle)
        self.water_no_train_list, self.water_no_test_list =  data_split(self.water_no_list, ratio, shuffle)

        train_list = (self.water_train_list, self.water_no_train_list)
        test_list = (self.water_test_list, self.water_no_test_list)

        self._run(train_list, split='train')
        self._run(test_list, split='val')

        # save cls_labels_onehot
        np.save(os.path.join(self.npy_root, 'cls_labels_onehot.npy'), np.array(self.onehot_dict, dtype=object))
        print('::save cls_labels_onehot.npy done!')

    def run_unseen(self):

        test_list = (self.water_list, self.water_no_list)
        self._run(test_list, split='test')

        # save cls_labels_onehot
        np.save(os.path.join(self.npy_root, 'cls_labels_onehot.npy'), np.array(self.onehot_dict, dtype=object))
        print('::save cls_labels_onehot.npy done!')

def data_split(full_list, ratio, shuffle=True):
    """
    borrowed form https://blog.csdn.net/zichen_ziqi/article/details/105600397
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.seed(888)
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

if __name__ == '__main__':
    # process your own original data ['./PourIt/ori_scene1', './PourIt/ori_scene2', './PourIt/ori_scene3', ...]
    # For example
    SCENE_NUM = 3
    ori_data_root = ['./data/PourIt_additional/ori_scene{}'.format(i+1) for i in range(SCENE_NUM)]
    processor = Processor(ori_data_root=ori_data_root, pro_data_root='./data/PourIt_additional/seen', npy_root='./data/PourIt_additional/seen/')
    processor.run(ratio=0.9, shuffle=True)

    # processor = Processor(ori_data_root=['./PourIt/ori_data_v1', './PourIt/ori_data_v2', './PourIt/tap_1', './PourIt/tap_2', './PourIt/tap_3'], pro_data_root='./PourIt/pourit_v1v2tap', npy_root='./datasets/pourit/')
    # processor.run(ratio=0.9, shuffle=True)
    #
    # processor = Processor(ori_data_root=['/data/liquid/ori_data_v1'], pro_data_root='/data/liquid/v1', npy_root='./datasets/liquid/', have_labelImg=True)
    # processor.run(ratio=0.8, shuffle=True)
    #
    # processor = Processor(ori_data_root=['./PourIt/ori_data_v3'], npy_root='./datasets/pourit_unseen/', pro_data_root='./PourIt/pourit_unseen')
    # processor.run_unseen()




