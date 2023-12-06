'''
# -*- coding: utf-8 -*-
# @Project   : afa
# @File      : _init_path.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2022/8/28 19:02

# @Desciption: 
'''

import os
import sys

sys.path.insert(0, os.getcwd())

def add_path(path):
    if path not in sys.path: sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)
add_path(this_dir)

root_path = os.path.abspath(os.path.join(this_dir, '..'))
add_path(root_path)

root_path = os.path.abspath(os.path.join('..', 'yolov7'))
add_path(root_path)

