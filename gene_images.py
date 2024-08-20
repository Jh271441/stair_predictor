#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File Name: gene_images.py
@File Function: 
@Author: chenjunhao (chenjunhao07@baidu.com)
@Created Time: 2024/8/19
"""

import os

dirs = os.listdir("labelme_json_viz")
file_name = ['img.png', 'label.png', 'label_viz.png', 'label_names.txt']

target_dir = 'labelme_json_viz_processed'
target_sub_dir = ['images', 'labels', 'labels_viz', 'label_names']
for sub_dir in target_sub_dir:
    os.makedirs(os.path.join(target_dir, sub_dir), exist_ok=True)
for dir in dirs:
    file_dir_path = os.path.join("labelme_json_viz", dir)
    for i, f in enumerate(file_name):
        file_path = os.path.join(file_dir_path, f)
        os.system(f"cp {file_path} {os.path.join(target_dir, f'{target_sub_dir[i]}/{dir}_{f}')}")