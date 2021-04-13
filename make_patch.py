"""
@author:yebin
@file:make_patch.py
@IDE:Pycharm
@time:2020/12/17 下午1:07
@function:make_patch,a single white box, 一个epoch测试一次准确率
@example:
@tip:
"""
import argparse
import random
from utils import get_network, get_training_dataloader, get_test_dataloader, compute_mean_std
import os
import numpy as np
from conf import settings
from torch.autograd import Variable
from attack_utils import *
import torch.nn.functional as F
import torchvision.utils as vutils
import torch
from PIL import Image
import csv
import matplotlib.pyplot as plt
from GTSRB_dataloader import get_GTSRB_train_dataloader, get_GTSRB_test_dataloader

def save_patch(patch, save_path, epoch, batch_indx, adv_label):
    patch = patch[0].transpose(1, 2, 0)
    im = Image.fromarray(np.uint8(patch))
    im.save(os.path.join(save_path, "%d_%d_%d_patch.png" %(epoch, batch_indx, adv_label)))

def attack(x, patch, mask, epoch, indx):
    net.eval()

    x_out = F.softmax(net(x), dim=1)      #前向推理的结果 (1, 43)
    # x_ = x_out.data[:][target]
    target_prob = x_out.data[0][target]

    adv_x = torch.mul((1 - mask), x) + torch.mul(mask, patch)

    count = 0
    lr = args.lr
    while conf_target > target_prob and count < args.max_count:
        count += 1
        adv_x = Variable(adv_x.data, requires_grad=True)
        adv_out = F.log_softmax(net(adv_x), dim=1)     # (1, 43)

        adv_out_probs, adv_out_labels = adv_out.max(1)

        Loss = -adv_out[0][target]
        # print('%s_%d_loss:{:.3f}'.format(Loss) %("epoch", epoch))
        Loss.backward()

        adv_grad = adv_x.grad.clone()

        adv_x.grad.data.zero_()



        patch = patch - lr*adv_grad
        patch = torch.clamp(patch, min=-1, max=1)

        # test the patch
        adv_x = torch.mul((1 - mask), x) + torch.mul(mask, patch)
        # adv_x = torch.clamp(adv_x, min_out, max_out)   #TODO min_out, max_out待定
        adv_x = torch.clamp(adv_x, min=-1, max=1)
        out = F.softmax(net(adv_x), dim=1)
        target_prob = out.data[0][target]
        # y_argmax_prob = out.data.max(1)[0][0]

        print("%s_%d_[%d/840]count%d_target_prob:%f" %("epoch", epoch, indx, count, target_prob))
        # print(count, conf_target, target_prob, y_argmax_prob)
        #
        # if count >= args.max_count:
        #     break

    return adv_x, mask, patch


# def attack_bin(x, patch, mask, lr=0.01, epoch=30):
#     net.eval()
#
#     # x_out = F.softmax(net(x))
#
#     adv_x = torch.mul((1-mask), x) + torch.mul(mask, patch)
#
#     adv_x = Variable(adv_x.data, requires_grad=True)
#     adv_out = F.log_softmax(net(adv_x))
#
#     Loss = -adv_out[:, target].sum() / args.b
#     count = 0
#
#     while Loss > args.loss:
#         count += 1
#
#         adv_x = Variable(adv_x.data, requires_grad=True)
#         adv_out = F.log_softmax(net(adv_x))
#         Loss = -adv_out[:, target].sum() / args.b
#         print("%s%d-loss:{:.3f}".format(Loss) %("epoch", count))
#         Loss.backward()
#
#         if Loss < 1:
#             lr = 0.1
#         lr = lr
#         adv_grad = adv_x.grad.clone()
#         adv_x.grad.data.zero_()
#         patch -= lr * adv_grad
#
#         adv_x = torch.mul((1-mask), x) + torch.mul(mask, patch)
#
#     out = F.softmax(net(adv_x))
#     target_prob = out.data[:, target]
#
#     return adv_x, mask, patch


def train(epoch, patch, patch_shape):
    net.eval()
    success = 0
    total = 0

    for batch_indx, (images, labels) in enumerate(cifar100_training_loader):
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
        images, labels = Variable(images), Variable(labels)

        prediction = net(images)

        if prediction.data.max(1)[1][0] != labels.data[0] or labels.data[0] == target:  # TODO 考虑batchsize大于1的情况，暂时只做batchsize为1的情况
            continue

        #找出每个batch中的识别正确的总数
        # label_indx = torch.where((prediction.data.max(1)[1]==labels.data), torch.ones(args.b).cuda(), torch.zeros(args.b).cuda())
        # each_batch_sum = label_indx.cpu().numpy().sum()
        # total += each_batch_sum
        # indx = np.argwhere(label_indx.cpu().numpy()==1)
        # images = torch.stack([images.data[ind[0]] for ind in indx[:]])  #识别正确的图片
        # labels = torch.tensor([labels.data[ind[0]] for ind in indx[:]])
        total += 1

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()


        # transform patch
        data_shape = images.data.cpu().numpy().shape     #(1, 3, 32, 32)
        if patch_type == 'circle':
            patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size)
        elif patch_type == 'square':
            patch, mask, x_location, y_location = square_transform(patch, data_shape, patch_shape, image_size)
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)

        if args.gpu:
            patch, mask = patch.cuda(), mask.cuda()
        patch, mask = Variable(patch), Variable(mask)

        # adv_x, mask, patch = attack(images, patch, mask, epoch)
        adv_x, mask, patch = attack(images, patch, mask, epoch, batch_indx)

        adv_label = net(adv_x).data.max(1)[1][0]
        ori_label = labels[0]

        if adv_label == target:
            success += 1

            # if plot_all == 1:
            #     # 画出原图, 先做反归一化
            #     t_mean = torch.FloatTensor(settings.CIFAR100_TRAIN_MEAN).view(3, 1, 1).expand(3, 32, 32).cuda()
            #     t_std = torch.FloatTensor(settings.CIFAR100_TRAIN_STD).view(3, 1, 1).expand(3, 32, 32).cuda()
            #     ori_image_plot = images.data[0]*t_std + t_mean
            #
                # vutils.save_image(ori_image_plot, "./%s/%d_%d_%d_original.png" % (args.outf, epoch, batch_indx, ori_label),
            #                       normalize=False)
            #
            #
            #     adv_x_plot = adv_x.data[0]*t_std + t_mean
            #     # 画出对抗图像
            #     vutils.save_image(adv_x_plot, "./%s/%d_%d_%d_adversarial.png" % (args.outf, epoch, batch_indx, adv_label),
            #                       normalize=False)

        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()   #(3, 3, 32, 32)


        new_patch = patch[:, :, x_location:x_location + patch_shape[2], y_location:y_location + patch_shape[3]]

        # new_patch = np.zeros(patch_shape)        #(1, 3, 8, 8)
        # for i in range(new_patch.shape[0]):
        #     for j in range(new_patch.shape[1]):
        #         new_patch[i][j] = submatrix(patch[i][j])

        # t_mean = torch.FloatTensor(settings.CIFAR100_TRAIN_MEAN).view(3, 1, 1).expand(3, 8, 8)
        # t_std = torch.FloatTensor(settings.CIFAR100_TRAIN_STD).view(3, 1, 1).expand(3, 8, 8)
        # new_patch_image = torch.tensor(new_patch)
        # new_patch_image = new_patch_image*t_std + t_mean
        # new_patch_image = torch.clamp(new_patch_image, 0, 1)*255
        # new_patch_image = new_patch_image.numpy()

        patch = new_patch
        # np.save(os.path.join(args.patch_path, "%s_%d_patch.npy" %("epoch", epoch)), patch)

        # save_patch(new_patch_image, save_path=args.patch_path, epoch=epoch, batch_indx=batch_indx, adv_label=adv_label)

    # log to file
    progress_bar(batch_indx, len(cifar100_training_loader), "Train Patch Success: {:.3f}".format(success / total))
    print("Train Patch Success: {:.3f}".format(success / total))

    return patch, success / total


# def test(epoch, patch, patch_shape):
#     net.eval()
#     success = 0
#     total = 0
#
#     for batch_indx, (images, labels) in enumerate(cifar100_test_loader):
#         if args.gpu:
#             images = images.cuda()
#             labels = labels.cuda()
#         images, labels = Variable(images), Variable(labels)
#
#         prediction = net(images)
#
#         if prediction.data.max(1)[1][0] != labels.data[0] or (prediction.data.max(1)[1][0] == target):  # TODO 考虑batchsize大于1的情况，暂时只做batchsize为1的情况
#                                                          #排除本就识别错误的样本以及本来就是要攻击那个类的样本
#             continue
#
#         total += 1
#
#         # transform patch
#         data_shape = images.data.cpu().numpy().shape
#         if patch_type == 'circle':
#             patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size)
#
#         elif patch_type == 'square':
#             patch, mask = square_transform(patch, data_shape, patch_shape, image_size)
#
#         patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
#
#         if args.gpu:
#             patch, mask = patch.cuda(), mask.cuda()
#
#         patch, mask = Variable(patch), Variable(mask)
#
#         adv_x = torch.mul((1 - mask), images) + torch.mul(mask, patch)
#         # adv_x = torch.clamp(adv_x, min_out, max_out)
#         adv_x = torch.clamp(adv_x, -1, 1)
#
#         adv_label = net(adv_x).data.max(1)[1][0]
#         ori_label = labels.data[0]
#
#         if adv_label == target:
#             success += 1
#
#         masked_patch = torch.mul(mask, patch)
#         patch = masked_patch.data.cpu().numpy()
#
#         new_patch = np.zeros(patch_shape)
#         for i in range(new_patch.shape[0]):
#             for j in range(new_patch.shape[1]):
#                 new_patch[i][j] = submatrix(patch[i][j])
#
#         patch = new_patch
#
#     # log to file
#     progress_bar(batch_indx, len(cifar100_test_loader), "Test Success: {:.3f}".format(success / total))
#     print("%s_%d Test Success: {:.3f}".format(success / total) %("epoch", epoch))
#
#     return success / total




parser = argparse.ArgumentParser()
#
# parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
# parser.add_argument('--gpu', action='store_true', help='enables cuda')
#
# parser.add_argument('--target', type=int, default=859, help='The target class: 859 == toaster')
# parser.add_argument('--conf_target', type=float, default=0.9, help='Stop attack on image when target classifier reaches this value for target class')
#
# parser.add_argument('--max_count', type=int, default=1000, help='max number of iterations to find adversarial example')
# parser.add_argument('--patch_type', type=str, default='circle', help='patch type: circle or square')
# parser.add_argument('--patch_size', type=float, default=0.05, help='patch size. E.g. 0.05 ~= 5% of image ')
#
# parser.add_argument('--train_size', type=int, default=2000, help='Number of training images')
# parser.add_argument('--test_size', type=int, default=2000, help='Number of test images')
#
# parser.add_argument('--image_size', type=int, default=299, help='the height / width of the input image to network')
#
# parser.add_argument('--plot_all', type=int, default=1, help='1 == plot all successful adversarial images')
#
# parser.add_argument('--net', default='vgg16', help="The target classifier")
#
# parser.add_argument('--outf', default='./logs', help='folder to output images and model checkpoints')
# parser.add_argument('--manualSeed', type=int, default=1338, help='manual seed')
`# parser.add_argument('--weights', type=str, required=True, help='the weights file you want to test')
# parser.add_argument('--b', type=int, default=1, help='batch size for dataloader')
#
args = parser.parse_args()
args.epochs = 100
args.gpu = True
args.target = 5
args.conf_target = 0.9
args.patch_type = 'square'
args.train_size = 840
args.test_size = 420
args.image_size = 256
args.net = 'googlenet'
args.weights = '/data/user1/yebin/work/ModelClasstest/pytorch-cifar100/checkpoint/googlenet/Tuesday_29_December_2020_17h_57m_59s/googlenet-128-best.pth'
args.b = 1
args.outf = './logs/target'
args.plot_all = True
args.manualSeed = None
args.max_count = 500
args.patch_size = 0.05
args.lr = 1
args.loss = 0.1
args.patch_path = "/data/user1/yebin/work/ModelClasstest/pytorch-cifar100/train_res/img256/googlenet/mul_5%"
args.log_dir = args.patch_path + "/patch_attack_log.csv"
args.train_path = "/data/user1/yebin/work/Data_lib/GTSRB_原数据/patchdata/train"
args.test_path = "/data/user1/yebin/work/Data_lib/GTSRB_原数据/patchdata/test"
args.patch = None
try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.gpu:
    torch.cuda.manual_seed_all(args.manualSeed)

if torch.cuda.is_available() and not args.gpu:
    print("WARNING: You have a CUDA device, so you should probably run with --gpu")

target = args.target
conf_target = args.conf_target
max_count = args.max_count       #最大的迭代次数
patch_type = args.patch_type     #patch的形状
patch_size = args.patch_size
image_size = args.image_size
train_size = args.train_size     #训练图片的数量
test_size = args.test_size       #测试图片的数量
plot_all = args.plot_all         #绘制所有成功的对抗样本

print('===> 开始生成分类器，加载模型中 <===')
net = get_network(args.net, args.gpu)
net.load_state_dict(torch.load(args.weights))   #TODO:固定网络参数梯度，防止在训练中更新（目前尚未固定梯度）
print(net)
print('===> 分类器生成完毕！<===')

print('===>开始加载数据! <===')
cifar100_training_loader = get_GTSRB_train_dataloader(root=args.train_path,
        mean=settings.CIFAR100_TRAIN_MEAN,
        std=settings.CIFAR100_TRAIN_STD,
        batch_size=args.b,
        num_workers=0,
        shuffle=True
    )

cifar100_test_loader = get_GTSRB_test_dataloader(root=args.test_path,
        mean=settings.CIFAR100_TRAIN_MEAN,
        std=settings.CIFAR100_TRAIN_STD,
        batch_size=args.b,
        num_workers=0,
        shuffle=True
    )

print('===> 加载数据完成! <===')
# for batch_indx, (labels, images) in enumerate(cifar100_training_loader):
#     min_in = min(images.data)
#     max_in = max(images.data)

# min_in, max_in = net.input_range[0], net.input_range[1]    #TODO 暂不清楚
# min_in, max_in = np.array([min_in, min_in, min_in]), np.array([max_in, max_in, max_in])
# mean, std = np.array(net.mean), np.array(net.std)
# min_out, max_out = np.min((min_in-mean)/std), np.max((max_in-mean)/std)

if __name__ == '__main__':
    if patch_type == 'circle':
        patch, patch_shape = init_patch_circle(image_size, patch_size)

    elif patch_type == 'square':
        if args.patch:
            print("-----------！！！start from chechpoint！！！--------------")
            patch = np.load(args.patch)
 `           patch_shape = patch.shape
        else:
            patch, patch_shape = init_patch_square(image_size, patch_size)
    else:
        sys.exit("Please choose a square or circle patch")

    with open(args.log_dir, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_success", "test_success", "train_singel_success"])


    best_patch_epoch, best_patch_success_rate = 0, 0
    mean, std = settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD
    print("mean:", mean, "std:", std)
    for epoch in range(1, args.epochs + 1):
        patch, train_singel_success = train(epoch, patch, patch_shape)
        # np.save(os.path.join(args.patch_path, "%s_%d_patch.npy" % ("epoch", epoch)), patch)
        # plt.imshow(np.clip(np.transpose(patch[0], (1, 2, 0)) * std + mean, 0, 1))
        # plt.savefig("/home/yebin/work/ModelClasstest/pytorch-cifar100/train_res/5%noise/patch/" + str(epoch) + " patch.png")
        train_success_rate = test_patch(patch_type, target, patch, cifar100_training_loader, net, gpu=args.gpu, patch_shape=patch_shape, image_size=args.image_size)
        print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch, 100 * train_success_rate))
        test_success_rate = test_patch(patch_type, target, patch, cifar100_test_loader, net, gpu=args.gpu, patch_shape=patch_shape, image_size=args.image_size)
        print("Epoch:{} Patch attack success rate on testset: {:.3f}%".format(epoch, 100 * test_success_rate))
        np.save(os.path.join(args.patch_path + "/patch/", "%s_%d_patch.npy" % ("epoch", epoch)), patch)
        plt.imshow(np.clip(np.transpose(patch[0], (1, 2, 0)) * std + mean, 0, 1))
        plt.savefig(
            args.patch_path + "/patch/" + str(epoch) + " patch.png")


        with open(args.log_dir, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_success_rate, test_success_rate, train_singel_success])

        if test_success_rate > best_patch_success_rate:
            best_patch_success_rate = test_success_rate
            best_patch_epoch = epoch
            plt.imshow(np.clip(np.transpose(patch[0], (1, 2, 0)) * std + mean, 0, 1))
            plt.savefig(args.patch_path + "/patch/best_patch.png")

    print("The best patch is found at epoch {} with success rate {}% on testset".format(best_patch_epoch,
                                                                                        100 * best_patch_success_rate))
    # patch = train(args.epochs, patch, patch_shape)
