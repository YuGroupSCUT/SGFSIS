import os
import os.path
import cv2
import numpy as np
import glob
from torch.utils.data import Dataset
import torch
import pickle
import scipy.io as sio
import random
import torchvision.transforms as trans
from PIL import Image
import pandas as pd
from scipy.ndimage import gaussian_filter

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

class QuadRandomFlip():
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, img, mask1=None, mask2=None, mask3=None, mask4=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask1 is not None:
                mask1 = cv2.flip(mask1, d)
            if mask2 is not None:
                mask2 = cv2.flip(mask2, d)
            if mask3 is not None:
                mask3 = cv2.flip(mask3, d)
            if mask3 is not None:
                mask4 = cv2.flip(mask4, d)
        return img, mask1, mask2, mask3, mask4

class QuadRandomRotate():
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, img, mask1=None, mask2=None, mask3=None, mask4=None):
        if random.random() < self.prob:
            angle = random.randint(0, 4)
            img = np.rot90(img, angle)
            if mask1 is not None:
                mask1 = np.rot90(mask1, angle)
            if mask2 is not None:
                mask2 = np.rot90(mask2, angle)
            if mask3 is not None:
                mask3 = np.rot90(mask3, angle)
            if mask4 is not None:
                mask4 = np.rot90(mask4, angle)
        return img, mask1, mask2, mask3, mask4

class QuadCompose():
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img, mask1=None, mask2=None, mask3=None, mask4=None):
        for t in self.transforms:
            img, mask1, mask2, mask3, mask4 = t(img, mask1, mask2, mask3, mask4)
        return img, mask1, mask2, mask3, mask4


def get_gaussian_map_from_points(points_map, radius):
    if np.sum(points_map > 0):
        label_detect = gaussian_filter(points_map.astype(np.float64), sigma=radius/3)
        val = np.min(label_detect[points_map > 0])
        label_detect = label_detect / val
        label_detect[label_detect < 0.05] = 0
        label_detect[label_detect > 1] = 1
    else:
        label_detect = np.zeros(points_map.shape)
    return label_detect

class EXT_Metatrain_Dataset(Dataset):
    def __init__(self,
                 ext_root,
                 transform=None,
                 mag='40x',
                 dataset_name='consep'):

        self.name = dataset_name
        self.mag = mag
        self.img_list = sorted(glob.glob(os.path.join(ext_root, 'Images', '*.png')))
        self.label_list = sorted(glob.glob(os.path.join(ext_root, 'Label_cr_1_ct_1', '*.mat')))

        # 检查数据是否对的上
        for img_path, label_path in zip(self.img_list, self.label_list):
            assert os.path.basename(img_path).split('.')[0] == \
                   os.path.basename(label_path).split('.')[0], "图像与标签对应不上！"

        self.transform = transform
        self.totensor = trans.ToTensor()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        image_path, label_path = self.img_list[index], self.label_list[index]

        if not os.path.exists(image_path) or not os.path.exists(label_path):
            print('{} does not exist.'.format(image_path))
        image = Image.open(image_path).convert('RGB')
        label_cls = sio.loadmat(label_path)['cls_map']
        if self.name == 'pannuke':
            label_cls[label_cls == 5] = 1
        label_center = sio.loadmat(label_path)['center_map']
        label_contour = sio.loadmat(label_path)['contour_map']
        label_inst = sio.loadmat(label_path)['inst_map']
        raw_label_inst = np.copy(label_inst)


        if self.transform is not None:
            image, label_cls, label_contour, label_center, label_inst = self.transform(image,
                                                                                       label_cls,
                                                                                       label_contour,
                                                                                       label_center,
                                                                                       label_inst)
        else:
            image = np.array(image)

        # expand the contour
        label_contour[label_contour != 0] = 1
        if self.mag == '40x':
            kernel = np.ones((3, 3), np.uint8)
            label_contour = cv2.dilate(label_contour, kernel, iterations=1)

        # expand the centroid
        label_center[label_center != 0] = 1
        # get gaussian map of centroid map
        if self.mag == '40x':
            kernel = np.ones((5, 5), np.uint8)
            label_center_gaussian = get_gaussian_map_from_points(label_center, radius=8)
        elif self.mag == '20x':
            kernel = np.ones((3, 3), np.uint8)
            label_center_gaussian = get_gaussian_map_from_points(label_center, radius=6)
        # kernel = np.ones((3, 3), np.uint8)
        label_center = cv2.dilate(label_center, kernel, iterations=1)

        label_fore = np.array(label_cls > 0, dtype=np.int32)
        image = self.totensor(image)

        return image, \
               torch.Tensor(label_cls), \
               torch.Tensor(label_fore), \
               torch.Tensor(raw_label_inst), \
               torch.Tensor(label_contour), \
               torch.tensor(label_center), \
               torch.tensor(label_center_gaussian)


class Tsk_Finetune_Dataset(Dataset):
    def __init__(self,
                 tsk_source,
                 tsk_root,
                 shot=5,
                 run=0,
                 mode='train',
                 transform=None,
                 mag='40x'):

        self.name = tsk_source
        self.mag = mag
        self.tsk_img_path_list = sorted(glob.glob(os.path.join(tsk_root, 'Images', '*.png')))
        self.tsk_label_path_list = sorted(glob.glob(os.path.join(tsk_root, 'Label_cr_1_ct_1', '*.mat')))

        # 检查数据是否对的上
        for img_path, label_path in zip(self.tsk_img_path_list, self.tsk_label_path_list):
            assert os.path.basename(img_path).split('.')[0] == \
                   os.path.basename(label_path).split('.')[0], "图像与标签对应不上！"

        if mode == 'train':
            with open(f'./rand_split_file_{tsk_source}/labeled_shot_{shot}/split_{run}.pkl','rb') as f:
                labeled_idx = pickle.load(f)['labeled']
                self.img_list = [self.tsk_img_path_list[i] for i in labeled_idx]
                self.label_list = [self.tsk_label_path_list[i] for i in labeled_idx]
        else:
            self.img_list = self.tsk_img_path_list
            self.label_list = self.tsk_label_path_list

        self.transform = transform
        self.totensor = trans.ToTensor()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        image_path, label_path = self.img_list[index], self.label_list[index]

        if not os.path.exists(image_path) or not os.path.exists(label_path):
            print('{} does not exist.'.format(image_path))
        image = Image.open(image_path).convert('RGB')
        label_cls = sio.loadmat(label_path)['cls_map']

        if self.name == 'pannuke':
            label_cls[label_cls == 5] = 1

        label_contour = sio.loadmat(label_path)['contour_map']
        label_center = sio.loadmat(label_path)['center_map']
        label_inst = sio.loadmat(label_path)['inst_map']
        raw_label_inst = np.copy(label_inst)

        if self.transform is not None:
            image, label_cls, label_contour, label_center, label_inst = self.transform(image,
                                                                                       label_cls,
                                                                                       label_contour,
                                                                                       label_center,
                                                                                       label_inst)
        else:
            image = np.array(image)

        # expand the contour
        label_contour[label_contour != 0] = 1
        if self.mag == '40x':
            kernel = np.ones((3, 3), np.uint8)
            label_contour = cv2.dilate(label_contour, kernel, iterations=1)

        # expand the centroid
        label_center[label_center != 0] = 1
        # get gaussian map of centroid map
        if self.mag == '40x':
            kernel = np.ones((5, 5), np.uint8)
            label_center_gaussian = get_gaussian_map_from_points(label_center, radius=8)
        elif self.mag == '20x':
            kernel = np.ones((3, 3), np.uint8)
            label_center_gaussian = get_gaussian_map_from_points(label_center, radius=6)
        # kernel = np.ones((3, 3), np.uint8)
        label_center = cv2.dilate(label_center, kernel, iterations=1)

        label_fore = np.array(label_cls > 0, dtype=np.int32)
        image = self.totensor(image)

        return image, \
               torch.Tensor(label_cls.astype(np.int32)), \
               torch.Tensor(label_fore), \
               torch.Tensor(raw_label_inst.astype(np.int32)), \
               torch.Tensor(label_contour), \
               torch.tensor(label_center), \
               torch.tensor(label_center_gaussian)

if __name__ == '__main__':
    dataset = Tsk_Finetune_Dataset(tsk_source='monusac', tsk_root='/home/data1/my/dataset/monusac/extracted_mirror/train/256x256_256x256/',
                                   shot=1, run=1, mode='train')
    for s in dataset:
        pass
    print(len(dataset))