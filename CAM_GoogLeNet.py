"""
可视化全连接层输出与输入映射到原图的热力图
for GoogLeNet
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader, get_training_dataloader
from GTSRB_dataloader import get_GTSRB_test_dataloader
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision import transforms

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-net', type=str, required=True, help='net type')
    # parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    # parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    # parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    # parser.add_argument('-root', type=str, required=True, help='the data path')
    # parser.add_argument('-feature_path', type=str, help='the save path of feature map')
    args = parser.parse_args()
    args.net = "googlenet"
    args.weights = "/data/user1/yebin/work/ModelClasstest/pytorch-cifar100/checkpoint/googlenet/Tuesday_29_December_2020_17h_57m_59s/googlenet-128-best.pth"
    args.gpu = True
    args.b = 1
    args.root = "/data/user1/yebin/work/Data_lib/CAM/GoogLeNet/image_patch"
    args.visual_heatmap = True
    img_path = "/data/user1/yebin/work/Data_lib/CAM/GoogLeNet/image_patch/0_adImg.png"
    save_path = "/data/user1/yebin/work/Data_lib/CAM/GoogLeNet/heatmap/"

    net = get_network(args.net, args.gpu)
    # for 32by32
    # cifar100_test_loader = get_test_dataloader(
    #     settings.CIFAR100_TRAIN_MEAN,
    #     settings.CIFAR100_TRAIN_STD,
    #     #settings.CIFAR100_PATH,
    #     num_workers=0,
    #     batch_size=args.b,
    # )
    t_mean = settings.CIFAR100_TRAIN_MEAN
    t_std = settings.CIFAR100_TRAIN_STD

    cifar100_test_loader = get_GTSRB_test_dataloader(root=args.root,
                                                     mean=settings.CIFAR100_TRAIN_MEAN,
                                                     std=settings.CIFAR100_TRAIN_STD,
                                                     batch_size=args.b,
                                                     num_workers=0,
                                                     shuffle=True
                                                     )

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    for n_iter, (image, label) in enumerate(cifar100_test_loader):
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

        if args.gpu:
            image = image.cuda()
            label = label.cuda()

        # 获取模型输出的feature
        output = net.prelayer(image)
        output = net.a3(output)
        output = net.b3(output)

        output = net.maxpool(output)

        output = net.a4(output)
        output = net.b4(output)
        output = net.c4(output)
        output = net.d4(output)
        output = net.e4(output)

        output = net.maxpool(output)

        output = net.a5(output)
        features = net.b5(output)

        print("feature.shape:", features.shape)
        features1 = net.avgpool(features)
        features1 = net.dropout(features1)
        features1 = features1.view(features1.size()[0], -1)
        output = net.linear(features1)

        print("")

        # 获取中间梯度定义辅助函数

        # 预测得分最高的那一类对应的输出scroe
        pred = torch.argmax(output).item()
        print("prd:", pred)
        pred_class = output[:, pred]
        print("pred_class:", pred_class)
        features.retain_grad()

        pred_class.backward()  # 计算梯度
        grads = features.grad  # 获取梯度
        print("grads:", grads.shape)     #(1, 1024, 63, 63)

        pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

        # 此处batch_size为1，所以去掉第0维（batch size 维）
        pooled_grads = pooled_grads[0]
        features = features[0]

        print("features.shape:", features.shape)   #(1024, 63, 63)
        # 最后一层feature的通道数
        for i in range(1024):  # TODO 这步好像有问题
            features[i, ...] *= pooled_grads[i, ...]

        # 以下部分同Keras版实现
        heatmap = features.detach().cpu().numpy()
        heatmap = np.mean(heatmap, axis=0)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        # 可视化原始热力图
        if args.visual_heatmap:
            plt.matshow(heatmap)
            plt.savefig(save_path + "heatmap.png")

        img = cv2.imread(img_path)  # 用cv2加载原始图像
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
        cv2.imwrite(save_path + "heatmap_and_img.png", superimposed_img)  # 将图像保存到硬盘
