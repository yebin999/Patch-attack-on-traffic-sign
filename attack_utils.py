"""
@author:yebin
@file:attack_utils.py
@IDE:Pycharm
@time:2020/12/8 上午9:24
@function:attack所需要的部分组件
@example:
@tip:
"""
import os
import sys
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

from scipy.ndimage.interpolation import rotate

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 35.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(' ' + msg)
    L.append(' | Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def submatrix(arr):
    x, y = np.nonzero(arr)  # 用于得到数组array中非零元素的位置，返回值为元组
    # Using the smallest and largest x and y indices of nonzero elements,
    # we can find the desired rectangular bounds.
    # And don't forget to add 1 to the top bound to avoid the fencepost problem.
    return arr[x.min():x.max() + 1, y.min():y.max() + 1]


class ToSpaceBGR(object):
    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):
    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor


def init_patch_circle(image_size, patch_size):  # 初始化一块形状为圆的patch， patch_size是个比例值
    image_size = image_size ** 2
    noise_size = int(image_size * patch_size)
    radius = int(math.sqrt(noise_size / math.pi))  # 圆的半径
    patch = np.zeros((1, 3, radius * 2, radius * 2))
    for i in range(3):
        a = np.zeros((radius * 2, radius * 2))
        cx, cy = radius, radius  # The center of circle
        y, x = np.ogrid[-radius: radius, -radius: radius]
        index = x ** 2 + y ** 2 <= radius ** 2
        a[cy - radius:cy + radius, cx - radius:cx + radius][index] = np.random.rand()
        idx = np.flatnonzero((a == 0).all((1)))  #
        a = np.delete(a, idx, axis=0)
        patch[0][i] = np.delete(a, idx, axis=1)
    return patch, patch.shape


# 主要进行旋转变换
def circle_transform(patch, data_shape, patch_shape, image_size):  # data_shape应该是个4-D的，patch_shape也是4-D
    # get dummy image
    x = np.zeros(data_shape)

    # get shape
    m_size = patch_shape[-1]

    for i in range(x.shape[0]):

        # # random rotation
        # rot = np.random.choice(360)
        # for j in range(patch[i].shape[0]):
        #     patch[i][j] = rotate(patch[i][j], angle=rot, reshape=False)

        # random rotation
        rot = np.random.choice(360)
        for j in range(patch[0].shape[0]):
            patch[0][j] = rotate(patch[0][j], angle=rot, reshape=False)

        # random location
        random_x = np.random.choice(image_size)
        if random_x + m_size > x.shape[-1]:
            while random_x + m_size > x.shape[-1]:
                random_x = np.random.choice(image_size)
        random_y = np.random.choice(image_size)
        if random_y + m_size > x.shape[-1]:
            while random_y + m_size > x.shape[-1]:
                random_y = np.random.choice(image_size)

        # # apply patch to dummy image
        # x[i][0][random_x:random_x + patch_shape[-1], random_y:random_y + patch_shape[-1]] = patch[i][0]
        # x[i][1][random_x:random_x + patch_shape[-1], random_y:random_y + patch_shape[-1]] = patch[i][1]
        # x[i][2][random_x:random_x + patch_shape[-1], random_y:random_y + patch_shape[-1]] = patch[i][2]

        # apply patch to dummy image
        x[i][0][random_x:random_x + patch_shape[-1], random_y:random_y + patch_shape[-1]] = patch[0][0]
        x[i][1][random_x:random_x + patch_shape[-1], random_y:random_y + patch_shape[-1]] = patch[0][1]
        x[i][2][random_x:random_x + patch_shape[-1], random_y:random_y + patch_shape[-1]] = patch[0][2]

    mask = np.copy(x)
    mask[mask != 0] = 1.0

    return x, mask, patch.shape


def init_patch_square(image_size, patch_size):
    # get mask
    image_size = image_size ** 2
    noise_size = image_size * patch_size
    noise_dim = int(noise_size ** (0.5))
    patch = (np.random.rand(1, 3, noise_dim, noise_dim) - 0.5) / 0.5
    return patch, patch.shape


def square_transform(patch, data_shape, patch_shape, image_size):
    # get dummy image
    x = np.zeros(data_shape)

    # get shape
    m_size = patch_shape[-1]

    for i in range(x.shape[0]):

        # # random rotation
        # rot = np.random.choice(4)
        # for j in range(patch[i].shape[0]):
        #     patch[i][j] = np.rot90(patch[i][j], rot)

        # random rotation
        rot = np.random.choice(4)
        for j in range(patch[i].shape[0]):
            patch[0][j] = np.rot90(patch[i][j], rot)

        # random location
        random_x = np.random.choice(image_size)
        if random_x + m_size > x.shape[-1]:
            while random_x + m_size > x.shape[-1]:
                random_x = np.random.choice(image_size)

        random_y = np.random.choice(image_size)
        if random_y + m_size > x.shape[-1]:
            while random_y + m_size > x.shape[-1]:
                random_y = np.random.choice(image_size)


        # # apply patch to dummy image
        # x[i][0][random_x:random_x + patch_shape[-1], random_y:random_y + patch_shape[-1]] = patch[i][0]
        # x[i][1][random_x:random_x + patch_shape[-1], random_y:random_y + patch_shape[-1]] = patch[i][1]
        # x[i][2][random_x:random_x + patch_shape[-1], random_y:random_y + patch_shape[-1]] = patch[i][2]

        # apply patch to dummy image
        x[i][0][random_x:random_x + patch_shape[-1], random_y:random_y + patch_shape[-1]] = patch[i][0]
        x[i][1][random_x:random_x + patch_shape[-1], random_y:random_y + patch_shape[-1]] = patch[i][1]
        x[i][2][random_x:random_x + patch_shape[-1], random_y:random_y + patch_shape[-1]] = patch[i][2]

    mask = np.copy(x)
    mask[mask != 0] = 1.0

    return x, mask, random_x, random_y


def test_patch(patch_type, target, patch, test_loader, model, gpu, patch_shape, image_size):
    model.eval()
    test_total, test_actual_total, test_success = 0, 0, 0
    for batch_in, (images, labels) in enumerate(test_loader):
        if gpu:
            images = images.cuda()
            labels = labels.cuda()

        images, labels = Variable(images), Variable(labels)
        prediction = model(images)

        if prediction.data.max(1)[1][0] != labels.data[0] or (labels.data[0] == target):
            continue

        test_actual_total += 1

        data_shape = images.data.cpu().numpy().shape
        if patch_type == 'circle':
            applied_patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size=image_size)

        elif patch_type == 'square':
            applied_patch, mask, random_x, random_y = square_transform(patch, data_shape, patch_shape, image_size=image_size)

        applied_patch = torch.FloatTensor(applied_patch)
        mask = torch.FloatTensor(mask)

        if gpu:
            applied_patch, mask = applied_patch.cuda(), mask.cuda()

        applied_patch, mask = Variable(applied_patch), Variable(mask)

        adv_x = torch.mul((1 - mask), images) + torch.mul(mask, applied_patch)
        adv_label = model(adv_x).data.max(1)[1][0].cpu()
        if adv_label == target:
            test_success += 1

    return test_success/test_actual_total

def test_ensemble_patch(patch_type, target, patch, test_loader, model1, model2, model3, gpu, patch_shape, image_size):
    model1.eval()
    model2.eval()
    model3.eval()
    test_total, test_actual_total, test_success1, test_success2, test_success3, test_ensemble_success = 0, 0, 0, 0, 0, 0
    for batch_in, (images, labels) in enumerate(test_loader):
        if gpu:
            images = images.cuda()
            labels = labels.cuda()

        images, labels = Variable(images), Variable(labels)
        prediction1 = model1(images)
        prediction2 = model2(images)
        prediction3 = model3(images)

        if prediction1.data.max(1)[1][0] != labels.data[0] or prediction2.data.max(1)[1][0] != labels.data[0] or prediction3.data.max(1)[1][0] != labels.data[0] or (labels.data[0] == target):
            continue

        test_actual_total += 1

        data_shape = images.data.cpu().numpy().shape
        if patch_type == 'circle':
            applied_patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size=image_size)

        elif patch_type == 'square':
            applied_patch, mask, random_x, random_y = square_transform(patch, data_shape, patch_shape, image_size=image_size)

        applied_patch = torch.FloatTensor(applied_patch)
        mask = torch.FloatTensor(mask)

        if gpu:
            applied_patch, mask = applied_patch.cuda(), mask.cuda()

        applied_patch, mask = Variable(applied_patch), Variable(mask)

        adv_x = torch.mul((1 - mask), images) + torch.mul(mask, applied_patch)
        adv_label1 = model1(adv_x).data.max(1)[1][0].cpu()
        adv_label2 = model2(adv_x).data.max(1)[1][0].cpu()
        adv_label3 = model3(adv_x).data.max(1)[1][0].cpu()
        if adv_label1 == target:
            test_success1 += 1
        if adv_label2 == target:
            test_success2 += 1
        if adv_label3 == target:
            test_success3 += 1
        if adv_label1 == target and adv_label2 == target and adv_label3 == target:
            test_ensemble_success += 1

    return [test_success1/test_actual_total, test_success2/test_actual_total, test_success3/test_actual_total, test_ensemble_success/test_actual_total]
