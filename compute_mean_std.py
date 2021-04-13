"""
@author:yebin
@file:解析cifar100.py
@IDE:Pycharm
@time:2020/12/9 下午3:19
@function:计算cifar格式数据集的rgb各通道的mean和std值
@example:
@tip:
"""
import numpy
import pickle
import os


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    # data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    # data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    # data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    # mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    # std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)
    r = numpy.array([cifar100_dataset[i][:1024] for i in range(len(cifar100_dataset))])/255
    g = numpy.array([cifar100_dataset[i][1024:2048] for i in range(len(cifar100_dataset))])/255
    b = numpy.array([cifar100_dataset[i][2048:] for i in range(len(cifar100_dataset))])/255
    mean = numpy.mean(r), numpy.mean(g), numpy.mean(b)
    std = numpy.std(r), numpy.std(g), numpy.std(b)

    return mean, std

def unpickle(path):
    with open(os.path.join(path, 'test'), 'rb') as cifar100:
        cifar = pickle.load(cifar100, encoding='bytes')
        return cifar

d = unpickle("/home/yebin/work/ModelClasstest/pytorch-cifar100/GTSRB")

mean, std = compute_mean_std(d['data'])
print(mean, std)