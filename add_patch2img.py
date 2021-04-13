import numpy as np
import argparse
from GTSRB_dataloader import get_GTSRB_test_dataloader
from conf import settings
import torch
from PIL import Image
import os



def add_target_image(clean_image, patch, clean_image_size):    # clean_image, patch = tensor, numpy
    clean_image = clean_image.cpu()
    c_t_image = clean_image.clone()    # 1*3*H*W

    t_size = patch.shape[-1]    # H

    for i in range(clean_image.shape[0]):
        rot = np.random.choice(4)
        for j in range(patch[i].shape[0]):
            patch[0][j] = np.rot90(patch[i][j], rot)

    target_image = torch.from_numpy(patch)
    # random location
    # random_x = np.random.choice(clean_image_size)
    # while(random_x + t_size > clean_image_size):
    #     random_x = np.random.choice(clean_image_size)
    #
    # random_y = np.random.choice(clean_image_size)
    # while(random_y + t_size > clean_image_size):
    #     random_y = np.random.choice(clean_image_size)
    random_x = 0
    random_y = 0

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
    args = parser.parse_args()
    args.patch = "/data/user1/yebin/work/Data_lib/CAM/GoogLeNet/patch/epoch_41_patch.npy"
    args.root = "/data/user1/yebin/work/Data_lib/CAM/GoogLeNet/ori_image"
    args.save_path = "/data/user1/yebin/work/Data_lib/CAM/GoogLeNet/image_patch"
    args.gpu = True
    args.b = 1
    t_mean = settings.CIFAR100_TRAIN_MEAN
    t_std = settings.CIFAR100_TRAIN_STD

    cifar100_test_loader = get_GTSRB_test_dataloader(root=args.root,
                                                     mean=settings.CIFAR100_TRAIN_MEAN,
                                                     std=settings.CIFAR100_TRAIN_STD,
                                                     batch_size=args.b,
                                                     num_workers=0,
                                                     shuffle=True
                                                     )
    #加载patch
    patch = np.load(args.patch)
    patch_shape = patch.shape         # patch.shape:1*C*H*W

    for n_iter, (image, label) in enumerate(cifar100_test_loader):
        if args.gpu:
            image = image.cuda()
            label = label.cuda()
        # label = target_label

        c_t_image = add_target_image(image, patch, clean_image_size=256)
        save_ct_image(n_iter, args.save_path, c_t_image)
