from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from filter import FastGuidedFilter


def get_image(path, height=None, width=None, flag=False):
    if flag is True:
        image = Image.open(path).convert('RGB') #imread(path, mode='RGB')
    else:
        image = Image.open(path).convert('L')

    if height is not None and width is not None:
        image = np.array(image.resize((height, width)))
    else:
        image = np.array(image)

    return image


def get_single_dolp_image(path):
    DOP = get_image(path).astype(np.float32)
    img = DOP
    minner = np.amin(img)
    img = img - minner
    maxer = np.amax(img)
    img = img / maxer
    img = img.astype(np.float32)
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img).unsqueeze(0)  # Shape: [1, C, H, W]
    DOP_d = img

    return DOP_d


def interpolate_image(img, H, W):
    return F.interpolate(img, size=(H,W))


def get_coords(H, W):
    coords = np.dstack(np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W)))
    coords = torch.from_numpy(coords).float().cuda()
    return coords


def get_patches(img, KERNEL_SIZE):
    kernel = torch.zeros((KERNEL_SIZE ** 2, 1, KERNEL_SIZE, KERNEL_SIZE)).cuda()

    ks = KERNEL_SIZE
    sigma = 3
    center = ks // 2
    x = torch.arange(ks) - center
    y = torch.arange(ks) - center
    X, Y = torch.meshgrid(x, y, indexing='ij')
    squared_dist = X ** 2 + Y ** 2
    A = torch.exp(-squared_dist / (2 * sigma))

    for i in range(KERNEL_SIZE):
        for j in range(KERNEL_SIZE):
            kernel[int(torch.sum(kernel).item()),0,i,j] = A[i,j]

    pad = nn.ReflectionPad2d(KERNEL_SIZE//2)
    im_padded = pad(img)

    extracted = torch.nn.functional.conv2d(im_padded, kernel, padding=0).squeeze(0)

    return torch.movedim(extracted, 0, -1)


def filter_up(x_lr, y_lr, x_hr, r=1):
    guided_filter = FastGuidedFilter(r=r)
    y_hr = guided_filter(x_lr, y_lr, x_hr)
    y_hr = torch.clip(y_hr, 0, 1)
    return y_hr