"""
@author:yebin
@file:patch_utils.py
@IDE:Pycharm
@time:2020/12/17 下午5:13
@function:暂时没有用到
@example:
@tip:
"""
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def test_patch(patch_type, target, patch, test_loader, model, gpu):
    model.eval()
    test_total, test_actual_total, test_success = 0, 0, 0
    for batch_in, (images, labels) in enumerate(test_loader):
        if gpu:
            images = images.cuda()
            labels = labels.cuda()

        images, labels = Variable(images), Variable(labels)
        prediction = model(images)

        if prediction.data.max(1)[1] != labels.data[0] and (labels.data[0] == target):
            continue

        test_actual_total += 1

        data_shape = images.data.cpu().numpy().shape
        if patch_type == 'circle':
            patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size)

        elif patch_type == 'square':
            patch, mask = square_transform(patch, data_shape, patch_shape, image_size)