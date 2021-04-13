"""
@author:yebin
@file:test_target.py
@time:2021/1/11 上午10:34
@function:add the target image to clean image and then make the testing of  attack success rate
@example:
@tip:
"""
import argparse

from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader, get_training_dataloader
from GTSRB_dataloader import get_GTSRB_test_dataloader
from PIL import Image
import os






def add_target_image(clean_image, target_image, clean_image_size):    # clean_image, target_image = tensor, tensor
    clean_image = clean_image.cpu()
    target_image = target_image.cpu().numpy()

    c_t_image = clean_image.clone()    # 1*3*H*W

    t_size = target_image.shape[-1]    # H

    for i in range(clean_image.shape[0]):
        rot = np.random.choice(4)
        for j in range(target_image[i].shape[0]):
            target_image[0][j] = np.rot90(target_image[i][j], rot)

    target_image = torch.from_numpy(target_image)
    # random location
    random_x = np.random.choice(clean_image_size)
    while(random_x + t_size > clean_image_size):
        random_x = np.random.choice(clean_image_size)

    random_y = np.random.choice(clean_image_size)
    while(random_y + t_size > clean_image_size):
        random_y = np.random.choice(clean_image_size)

    c_t_image[:target_image.shape[0], :target_image.shape[1], random_x:target_image.shape[2]+random_x, random_y:target_image.shape[3]+random_y].copy_(target_image)

    return c_t_image

def save_ct_image(indx, save_path, image):   #image tensor=(1, 3, 256, 256)
    std = (0.5, 0.5, 0.5)
    mean = (0.5, 0.5, 0.5)
    img = image.cpu().numpy()
    img = np.clip(np.transpose(img[0], (1, 2, 0)) * std + mean, 0, 1) * 255
    img = Image.fromarray(np.uint8(img))
    img.save(os.path.join(save_path, "%d_adImg.png" %(indx)))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-net', type=str, required=True, help='net type')
    # parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    # parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    # parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    # parser.add_argument('-root', type=str, required=True, help='the data path')
    args = parser.parse_args()
    args.gpu = True
    args.b = 1
    args.root = "/data/user1/yebin/work/Data_lib/GTSRB_原数据/patchdata/train"
    save_path = "/data/user1/yebin/work/Data_lib/GTSRB_原数据/googlenet/adImg_train"

    # vgg16
    # args.weights = "/home/yebin/work/ModelClasstest/pytorch-cifar100/checkpoint/vgg16/GTSRB256to256/vgg16-191-best.pth"
    # args.net = 'vgg16'

    # for googlenet
    args.weights = "/data/user1/yebin/work/ModelClasstest/pytorch-cifar100/checkpoint/googlenet/Tuesday_29_December_2020_17h_57m_59s/googlenet-128-best.pth"
    args.net = 'googlenet'

    # for resnet34
    #args.weights = "/home/yebin/work/ModelClasstest/pytorch-cifar100/checkpoint/Resnet34/resnet34-121-best.pth"
    #args.net = 'resnet34'

    target_root = "/data/user1/yebin/work/Data_lib/targetImage"
    attack_target = 5
    net = get_network(args.net, args.gpu)
    # for 32by32
    # cifar100_test_loader = get_test_dataloader(
    #     settings.CIFAR100_TRAIN_MEAN,
    #     settings.CIFAR100_TRAIN_STD,
    #     #settings.CIFAR100_PATH,
    #     num_workers=0,
    #     batch_size=args.b,
    # )
    target_loader = get_GTSRB_test_dataloader(root=target_root,
                                                     mean=settings.CIFAR100_TRAIN_MEAN,
                                                     std=settings.CIFAR100_TRAIN_STD,
                                                     batch_size=1,
                                                     num_workers=0,
                                                     shuffle=True
                                                     )

    # for n_i, (t_image, t_label) in enumerate(target_loader):
    #     if args.gpu:
    #         target_image = t_image.cuda()
    #         target_label = t_label.cuda()
    #     else:
    #         target_image = t_image
    #         target_label = t_label

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

    with torch.no_grad():
        for n_i, (t_image, t_label) in enumerate(target_loader):
            if args.gpu:
                target_image = t_image.cuda()
                target_label = t_label.cuda()
            else:
                target_image = t_image
                target_label = t_label


        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
           # label = target_label

            c_t_image = add_target_image(image, target_image, clean_image_size=256)
            # save_ct_image(n_iter, save_path, c_t_image)
            output = net(c_t_image.cuda())
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()


    print()
    print("len:", len(cifar100_test_loader.dataset))
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
