# coding: utf-8
import os
import cv2 as cv
import numpy as np
from . import augmentation
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, aug_params=None, transform=None, if_test=False, cls_num=4):

        self.aug_params = aug_params
        self.transform = transform
        self.if_test = if_test
        self.cls_num = cls_num

    def label_statistic(self):
        cls_count = np.zeros(self.cls_num).astype(np.int64)
        for label in self.labels_list:
            cls_count[label] += 1
        for i in range(self.cls_num):
            print("Class {}: {}".format(str(i), cls_count[i]))
        print("Summary: {}".format(np.sum(cls_count)))
        return cls_count

    def label_weights_for_balance(self, C=100.0):
        cls_count = self.label_statistic()
        labels_weight_list = []
        for label in self.labels_list:
            labels_weight_list.append(C/float(cls_count[label]))
        return labels_weight_list


class MultiDataset(BaseDataset):
    """Multi-modal Dataset"""
    def __init__(
        self, pairs_path_list, labels_list=None,
        aug_params=None, transform=None, if_test=False, cls_num=4):
        super(MultiDataset, self).__init__(aug_params, transform, if_test, cls_num)
        self.pairs_path_list = pairs_path_list
        if not self.if_test:
            self.labels_list = labels_list
        
    def __getitem__(self, index):
        img_f_path, img_o_path = self.pairs_path_list[index]
        img_f_filename = os.path.split(img_f_path)[-1]
        img_f = (cv.imread(img_f_path)/255.).astype(np.float32)
        img_o_filename = os.path.split(img_o_path)[-1]
        img_o = (cv.imread(img_o_path)/255.).astype(np.float32)        

        if self.aug_params:
            aug = augmentation.OurAug(self.aug_params)
            img_f = aug.process(img_f)
            aug = augmentation.OurAug(self.aug_params)
            img_o = aug.process(img_o)
            
        if self.transform:
            img_f = self.transform(img_f)
            img_o = self.transform(img_o)

        if not self.if_test:
            label = self.labels_list[index]
            label_onehot = np.zeros(self.cls_num).astype(np.float32)
            label_onehot[label] = 1.
        else:
            label_onehot = -1

        return (img_f, img_o), label_onehot, (img_f_filename, img_o_filename)

    def __len__(self):
        return len(self.pairs_path_list)


class SingleDataset(BaseDataset):
    def __init__(
            self, imgs_path_list, labels_list=None,
            aug_params=None, transform=None, if_test=False, cls_num=4):
        super(SingleDataset, self).__init__(aug_params, transform, if_test, cls_num)
        self.imgs_path_list = imgs_path_list
        if not self.if_test:
            self.labels_list = labels_list

    def __getitem__(self, index):

        img_path = self.imgs_path_list[index]
        img_filename = os.path.split(img_path)[-1]
        img = (cv.imread(img_path) / 255.).astype(np.float32)

        if not self.if_test:
            label = self.labels_list[index]
            label_onehot = np.zeros(self.cls_num).astype(np.float32)
            label_onehot[label] = 1.0
        else:
            label_onehot = -1

        if self.aug_params is not None:
            myAug = augmentation.OurAug(self.aug_params)
            img = myAug.process(img)
        if self.transform:
            img = self.transform(img)

        return img, label_onehot, img_filename

    def __len__(self):
        return len(self.imgs_path_list)