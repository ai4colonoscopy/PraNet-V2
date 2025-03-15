#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import random
import numpy as np
import torch
from scipy import ndimage
# from scipy.ndimage.interpolation import zoom
from scipy.ndimage import zoom
from torch.utils.data import Dataset

import matplotlib.pyplot as plt


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # 找到非黑色区域的边界
        non_zero_coords = np.argwhere(image > -1)
        if non_zero_coords.size == 0:
            print("Empty image")

        min_coord = non_zero_coords.min(axis=0)
        max_coord = non_zero_coords.max(axis=0)
        # 裁剪图像和标签
        image_cropped = image[min_coord[0]:max_coord[0]+1, min_coord[1]:max_coord[1]+1]
        label_cropped = label[min_coord[0]:max_coord[0]+1, min_coord[1]:max_coord[1]+1]
        
        # visualize_data(image_cropped, label_cropped, 0,save_dir='./')
        # 缩放回原始尺寸
        zoom_factors = (
            self.output_size[0] / image_cropped.shape[0],  # 高度缩放比例
            self.output_size[1] / image_cropped.shape[1]   # 宽度缩放比例
        )
        # 使用双线性插值法放大图像
        image_zoomed = zoom(image_cropped, zoom_factors, order=1)  # 对图像使用双线性插值
        # 使用最近邻插值法放大标签
        label_zoomed = zoom(label_cropped, zoom_factors, order=0)  # 对标签使用最近邻插值
        
        if random.random() > 0.5:
            image_zoomed, label_zoomed = random_rot_flip(image_zoomed, label_zoomed)
        elif random.random() > 0.5:
            image_zoomed, label_zoomed = random_rotate(image_zoomed, label_zoomed)

        x, y = image_zoomed.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image_zoomed = zoom(image_zoomed, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label_zoomed = zoom(label_zoomed, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image_zoomed = torch.Tensor(image_zoomed.astype(np.float32)).unsqueeze(0)
        label_zoomed = torch.Tensor(label_zoomed.astype(np.float32))
        sample = {'image': image_zoomed, 'label': label_zoomed.long()}
        return sample

        # if random.random() > 0.5:
        #     image, label = random_rot_flip(image, label)
        # elif random.random() > 0.5:
        #     image, label = random_rotate(image, label)
        # x, y = image.shape
        # if x != self.output_size[0] or y != self.output_size[1]:
        #     image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
        #     label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # image = torch.Tensor(image.astype(np.float32)).unsqueeze(0)
        # label = torch.Tensor(label.astype(np.float32))
        # sample = {'image': image, 'label': label.long()}
        # return sample


class ACDCdataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train" or self.split == "valid":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, self.split, slice_name)
            data = np.load(data_path)
            image, label = data['img'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}".format(vol_name)
            data = np.load(filepath)
            image, label = data['img'], data['label']

        sample = {'image': image, 'label': label}
        if self.transform and self.split == "train":
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample



# 定义可视化函数
def visualize_data(image, label, slice_index,save_dir):
    """
    可视化给定切片索引的图像和标签
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    if len(image.shape)==2:
        image = image[np.newaxis, :, :]
        slice_index = 0  # 对于2D数据，只有一个切片
    if len(label.shape)==2:
        label = label[np.newaxis, :, :]
        slice_index = 0  # 对于2D数据，只有一个切片
    
    # 图像数据可视化
    axes[0].imshow(image[slice_index,:,:], cmap='gray')
    axes[0].set_title('Image Slice')
    axes[0].axis('off')
    
    # 标签数据可视化
    axes[1].imshow(label[slice_index, :, :], cmap='jet', alpha=0.5)
    axes[1].set_title('Label Slice')
    axes[1].axis('off')
    
    save_path = os.path.join(save_dir, f'slice_{slice_index}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    print(f"Saved slice {slice_index} to {save_path}")

if __name__ == '__main__':


    # 加载数据
    data_path = '/defaultShare/archive/zhuzixuan/cascade_dataset/ACDC/test/case_092_volume_ES.npz'
    data = np.load(data_path)
    image, label = data['img'], data['label']

    # 查看数据形状
    print(f"Image shape: {image.shape}")
    print(f"Label shape: {label.shape}")



    # 选择一个切片进行可视化
    slice_index = 2 # 选择中间切片
    visualize_data(image, label, slice_index,save_dir='./')