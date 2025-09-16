import torch

from utils import *
from loss import *
from network import Network

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from imageio import imread, imsave
import matplotlib.pyplot as plt
import einops
import time


# -------------------------------
parser = argparse.ArgumentParser('ZEROPDENet')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--ws', default=30, type=int, help='Window size')
parser.add_argument('--ps', default=5, type=int, help='Patch size')
parser.add_argument('--nn', default=50, type=int, help='Number of nearest neighbors to search')
parser.add_argument('--mm', default=10, type=int, help='Number of pixel banks to use for training')
args = parser.parse_args()

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda:0"
WINDOW_SIZE = args.ws
PATCH_SIZE = args.ps
NUM_NEIGHBORS = args.nn

transform = transforms.Compose([transforms.ToTensor()])


def construct_pixel_bank(input_img):

    pad_sz = WINDOW_SIZE // 2 + PATCH_SIZE // 2
    center_offset = WINDOW_SIZE // 2
    blk_sz = 64  # Block size for processing  64

    start_time = time.time()

    # Load the already noisy image
    img = input_img
    img = img.cuda()  # No extra dimension is added

    # Pad the image (F.pad requires a 4D tensor)
    img_pad = F.pad(img, (pad_sz, pad_sz, pad_sz, pad_sz), mode='reflect')
    # Extract patches by unfolding the image into sliding window patches
    img_unfold = F.unfold(img_pad, kernel_size=PATCH_SIZE, padding=0, stride=1)
    H_new = img.shape[-2] + WINDOW_SIZE
    W_new = img.shape[-1] + WINDOW_SIZE
    img_unfold = einops.rearrange(img_unfold, 'b c (h w) -> b c h w', h=H_new, w=W_new)

    num_blk_w = img.shape[-1] // blk_sz
    num_blk_h = img.shape[-2] // blk_sz
    is_window_size_even = (WINDOW_SIZE % 2 == 0)
    topk_list = []

    # Process each block
    for blk_i in range(num_blk_w):
        for blk_j in range(num_blk_h):
            start_h = blk_j * blk_sz
            end_h = (blk_j + 1) * blk_sz + WINDOW_SIZE
            start_w = blk_i * blk_sz
            end_w = (blk_i + 1) * blk_sz + WINDOW_SIZE

            sub_img_uf = img_unfold[..., start_h:end_h, start_w:end_w]
            sub_img_shape = sub_img_uf.shape

            if is_window_size_even:
                sub_img_uf_inp = sub_img_uf[..., :-1, :-1]
            else:
                sub_img_uf_inp = sub_img_uf

            patch_windows = F.unfold(sub_img_uf_inp, kernel_size=WINDOW_SIZE, padding=0, stride=1)
            patch_windows = einops.rearrange(
                patch_windows,
                'b (c k1 k2 k3 k4) (h w) -> b (c k1 k2) (k3 k4) h w',
                k1=PATCH_SIZE, k2=PATCH_SIZE, k3=WINDOW_SIZE, k4=WINDOW_SIZE,
                h=blk_sz, w=blk_sz
            )

            img_center = einops.rearrange(
                sub_img_uf,
                'b (c k1 k2) h w -> b (c k1 k2) 1 h w',
                k1=PATCH_SIZE, k2=PATCH_SIZE,
                h=sub_img_shape[-2], w=sub_img_shape[-1]
            )
            img_center = img_center[..., center_offset:center_offset + blk_sz, center_offset:center_offset + blk_sz]

            # Compute L2 distances and select the most similar patches
            #                     (1,49,1,64,64)   (1,49,1600,64,64)
            #                               (1,1600,64,64)
            # l2_dis = torch.sum(torch.matmul(img_center,patch_windows),1)
            # _, sort_indices = torch.topk(l2_dis, k=NUM_NEIGHBORS, largest=True, sorted=True, dim=-3)

            l2_dis = torch.sum((img_center - patch_windows) ** 2, dim=1)
            _, sort_indices = torch.topk(l2_dis, k=NUM_NEIGHBORS, largest=False, sorted=True, dim=-3)

            patch_windows_reshape = einops.rearrange(
                patch_windows,
                'b (c k1 k2) (k3 k4) h w -> b c (k1 k2) (k3 k4) h w',
                k1=PATCH_SIZE, k2=PATCH_SIZE, k3=WINDOW_SIZE, k4=WINDOW_SIZE
            )
            patch_center = patch_windows_reshape[:, :, patch_windows_reshape.shape[2] // 2, ...]
            topk = torch.gather(patch_center, dim=-3,
                                index=sort_indices.unsqueeze(1).repeat(1, 1, 1, 1, 1))  # 第二个是 3
            topk_list.append(topk)

    # Merge results from all blocks to form the pixel bank
    topk = torch.cat(topk_list, dim=0)
    topk = einops.rearrange(topk, '(w1 w2) c k h w -> k c (w2 h) (w1 w)', w1=num_blk_w, w2=num_blk_h)
    topk = topk.permute(2, 3, 0, 1)

    elapsed = time.time() - start_time
    print(f"Processed img in {elapsed:.2f} seconds. Pixel bank shape: {topk.shape}")

    img_bank = topk.cpu().numpy().transpose((2, 0, 1, 3))
    # Use only the first mm banks for training
    img_bank = img_bank[:args.mm]
    img_bank = torch.from_numpy(img_bank).to(device)

    return img_bank

def get_pairImg_fromBank(img_bank):
    N, H, W, C = img_bank.shape

    index1 = torch.randint(0, N, size=(H, W), device=device)
    index1_exp = index1.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    img1 = torch.gather(img_bank, 0, index1_exp)  # Shape: (1, H, W, C)
    img1 = img1.permute(0, 3, 1, 2)  # (1, C, H, W)

    index2 = torch.randint(0, N, size=(H, W), device=device)
    eq_mask = (index2 == index1)
    if eq_mask.any():
        index2[eq_mask] = (index2[eq_mask] + 1) % N
    index2_exp = index2.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    img2 = torch.gather(img_bank, 0, index2_exp)
    img2 = img2.permute(0, 3, 1, 2)

    return img1, img2


def main(img_path,sav_path):
    DOP= get_single_dolp_image(img_path)
    img_bank = construct_pixel_bank(DOP)

    start_time = time.time()

    model = Network()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    DOP = DOP.to(device)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        img1, img2 = get_pairImg_fromBank(img_bank)
        _, img_v_fixed, loss = model(DOP, img1, img2)

        loss.backward()
        optimizer.step()

    model.eval()
    model.zero_grad()
    _, img_v_fixed, loss = model(DOP, DOP, DOP)
    img_v_fixed = img_v_fixed.cpu().squeeze(0).permute(1, 2, 0)
    out = img_v_fixed.squeeze(2).detach().numpy()
    out = (out - np.min(out)) / (np.max(out) - np.min(out))

    elapsed = time.time() - start_time
    print(f'时间：{elapsed:.2f}')

    imsave(sav_path, (out * 255).astype('uint8'))


if __name__ == '__main__':
    img_path = "./input/DOP.bmp"
    sav_path = './output/'+'DOP_pde.bmp'
    main(img_path, sav_path)
