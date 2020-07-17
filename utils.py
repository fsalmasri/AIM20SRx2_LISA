import os
import math
import numpy as np
from PIL import Image

from torchvision.transforms import ToTensor
import torch.utils.data as data

# def makeTails(npimLR, k= 400):
#     s = k //2
#     imglst = []
#     for i in range(0, npimLR.shape[0]-(k-s), s):
#         for j in range(0, npimLR.shape[1]-(k-s), s):
#             imglst.append(npimLR[i:i+k,j:j+k,:])
#     return imglst
#
#
# def reconstructImg(imglstHR, row, cln, k=400):
#     test = np.zeros((row * 2, cln * 2, 3))
#     c = 0
#     for i in range(0, test.shape[0] - k, k):
#         for j in range(0, test.shape[1] - k, k):
#             if i == 0:
#                 if j == 0:
#                     test[i:i + k * 2, j:j + k * 2] = imglstHR[c]
#                 if test[i:i + k * 2, j + k // 2:j + k * 2].shape == imglstHR[c][:, k // 2:].shape:
#                     test[i:i + k * 2, j + k // 2:j + k * 2] = imglstHR[c][:, k // 2:]
#             else:
#                 if j == 0:
#                     test[i:i + k * 2, j:j + k * 2] = imglstHR[c]
#                 if test[i + k // 2:i + k * 2, j + k // 2:j + k * 2].shape == imglstHR[c][k // 2:, k // 2:].shape:
#                     test[i + k // 2:i + k * 2, j + k // 2:j + k * 2] = imglstHR[c][k // 2:, k // 2:]
#             c = c + 1
#
#     return test

def make_tiles(img, tile_dim, overlap_dim):
    h, w, _ = img.shape
    o = []
    p_w, p_h = 0, 0

    N_w = math.ceil((w - overlap_dim) / (tile_dim - overlap_dim))
    N_h = math.ceil((h - overlap_dim) / (tile_dim - overlap_dim))
    change = False
    for i in range(N_w):
        for j in range(N_h):
            if p_w + tile_dim >= w:
                change = True
                p_w = w - tile_dim

            o += [img[p_h:p_h + tile_dim, p_w:p_w + tile_dim, :]]

            if change:
                change = False
                p_w = 0
                p_h += tile_dim - overlap_dim
                p_h = min(p_h, h - tile_dim)
            else:
                p_w += tile_dim - overlap_dim
    return o


def merge_tiles(tiles, overlap_dim, out_shape):
    tiles = np.stack(tiles, axis=0)

    d, _, _ = tiles[0].shape
    h, w, _ = out_shape

    o = np.zeros(out_shape, dtype=tiles.dtype)

    p_x, p_y = 0, 0
    change_x = False
    for tile in tiles:

        if p_x + d >= out_shape[1]:
            p_x = out_shape[1] - d
            change_x = True

        if p_y + d > out_shape[0]:
            p_y = out_shape[0] - d

        if p_y != 0:
            lim_y_inf = p_y + overlap_dim // 2
        else:
            lim_y_inf = p_y

        if p_y != h - d:
            lim_y_sup = p_y + d - overlap_dim // 2
        else:
            lim_y_sup = p_y + d

        if p_x != 0:
            lim_x_inf = p_x + overlap_dim // 2
        else:
            lim_x_inf = p_x

        if p_x != w - d:
            lim_x_sup = p_x + d - overlap_dim // 2
        else:
            lim_x_sup = p_x + d

        o[lim_y_inf:lim_y_sup, lim_x_inf:lim_x_sup, :] = tile[lim_y_inf - p_y:lim_y_sup - p_y,
                                                         lim_x_inf - p_x:lim_x_sup - p_x, :]
        # o[p_y:p_y + d,p_x:p_x+d,:] = tile # For complete update.

        if change_x:
            p_x = 0
            p_y += d - int(overlap_dim)
            change_x = False
        else:
            p_x += d - int(overlap_dim)
    return o



class AIM20(data.Dataset):
    def __init__(self, imgDir):
        self.img_dir = imgDir
        self.fileList = os.listdir(imgDir)

    def __getitem__(self, idx):
        name = self.fileList[idx]
        nameIn = os.path.join(self.img_dir, name)
        imgIn = Image.open(nameIn)
        return ToTensor()(imgIn), self.fileList[idx]

    def __len__(self):
        return len(self.fileList)