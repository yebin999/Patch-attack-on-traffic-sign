#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader, get_training_dataloader
from GTSRB_dataloader import get_GTSRB_test_dataloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-net', type=str, required=True, help='net type')
    # parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    # parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    # parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    # parser.add_argument('-root', type=str, required=True, help='the data path')
    args = parser.parse_args()
    args.net = "resnet34"
    args.weights = "/data/user1/yebin/work/ModelClasstest/pytorch-cifar100/checkpoint/resnet34/Saturday_26_December_2020_04h_00m_39s/resnet34-121-best.pth"
    args.gpu = True
    args.b = 1
    args.root = "/data/user1/yebin/work/ModelClasstest/pytorch-cifar100/physical_attack/single"
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
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
            label = torch.tensor([5]).cuda()
            output = net(image)
            index = output.data.max(1)[1][0]
            target_prob = F.softmax(output, dim=1)
            print("---------------------")
            print("n_iter:", n_iter)
            print("index:", index)
            print("target_prob:", target_prob.data[0][index])
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
