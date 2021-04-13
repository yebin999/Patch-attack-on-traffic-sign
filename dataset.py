""" train and test dataset

author baiyu
"""
import os
import sys
import pickle

from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset

class CIFAR100Train(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        #if transform is given, we transoform data using
        with open(os.path.join(path, 'train'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        # return len(self.data['fine_labels'.encode()])
        return len(self.data['labels'])

    def __getitem__(self, index):
        # label = self.data['fine_labels'.encode()][index]
        label = self.data['labels'][index]
        r = self.data['data'][index, :1024].reshape(32, 32)
        g = self.data['data'][index, 1024:2048].reshape(32, 32)
        b = self.data['data'][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return image, label

class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        with open(os.path.join(path, 'test'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['data'])

    def __getitem__(self, index):
        label = self.data['labels'][index]
        r = self.data['data'][index, :1024].reshape(32, 32)
        g = self.data['data'][index, 1024:2048].reshape(32, 32)
        b = self.data['data'][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return image, label

