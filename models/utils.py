from __future__ import print_function
from PIL import Image
from glob import glob
import random
import numpy as np
import collections
import math
import torch
import torch.nn as nn
from torch.autograd import Variable


######################################
# Functions
######################################


def cal_feat_mask(inMask, conv_layers, threshold):
    assert inMask.dim() == 4
    assert inMask.size(0) == 1
    inMask = inMask.float()
    convs = []
    inMask = Variable(inMask, requires_grad=False)
    for id_net in range(conv_layers):
        conv = nn.Conv2d(1, 1, 4, 2, 1, bias=False)
        conv.weight.data.fill_(1 / 16)
        convs.append(conv)
    lnet = nn.Sequential(*convs)
    if inMask.is_cuda:
        lnet = lnet.cuda()
    output = lnet(inMask)
    output = (output > threshold).float().mul_(1)
    output = Variable(output, requires_grad=False)
    return output.detach().byte()


def cal_mask_given_mask_thred(img, mask, patch_size, stride, mask_thred):
    assert img.dim() == 3
    assert mask.dim() == 2
    dim = img.dim()
    _, H, W = img.size(dim - 3), img.size(dim - 2), img.size(dim - 1)
    nH = int(math.floor((H - patch_size) / stride + 1))
    nW = int(math.floor((W - patch_size) / stride + 1))
    N = nH * nW
    flag = torch.zeros(N).long()
    offsets_tmp_vec = torch.zeros(N).long()
    nonmask_point_idx_all = torch.zeros(N).long()
    tmp_non_mask_idx = 0
    mask_point_idx_all = torch.zeros(N).long()
    tmp_mask_idx = 0

    for i in range(N):
        h = int(math.floor(i / nW))
        w = int(math.floor(i % nW))

        mask_tmp = mask[h * stride:h * stride + patch_size,
                   w * stride:w * stride + patch_size]
        if torch.sum(mask_tmp) < mask_thred:
            nonmask_point_idx_all[tmp_non_mask_idx] = i
            tmp_non_mask_idx += 1
        else:
            mask_point_idx_all[tmp_mask_idx] = i
            tmp_mask_idx += 1
            flag[i] = 1
            offsets_tmp_vec[i] = -1

    non_mask_num = tmp_non_mask_idx
    mask_num = tmp_mask_idx

    nonmask_point_idx = nonmask_point_idx_all.narrow(0, 0, non_mask_num)
    mask_point_idx = mask_point_idx_all.narrow(0, 0, mask_num)

    flatten_offsets_all = torch.LongTensor(N).zero_()
    for i in range(N):
        offset_value = torch.sum(offsets_tmp_vec[0:i + 1])
        if flag[i] == 1:
            offset_value = offset_value + 1
        flatten_offsets_all[i + offset_value] = -offset_value

    flatten_offsets = flatten_offsets_all.narrow(0, 0, non_mask_num)

    return flag, nonmask_point_idx, flatten_offsets, mask_point_idx


def cal_sps_for_Advanced_Indexing(h, w):
    sp_y = torch.arange(0, w).long()
    sp_y = torch.cat([sp_y] * h)
    lst = []
    for i in range(h):
        lst.extend([i] * w)
    sp_x = torch.from_numpy(np.array(lst))
    return sp_x, sp_y


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def info(object, spacing=10, collapse=1):
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print("\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]))


######################################
# Class
######################################

class Data_load(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform):
        super(Data_load, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.paths = glob('{:s}/*'.format(img_root),
                          recursive=True)
        self.mask_paths = glob('{:s}/*.png'.format(mask_root))
        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))
        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))
        return gt_img, mask

    def __len__(self):
        return len(self.paths)


class Pospara:
    def __init__(self):
        pass

    def update_output(self, input, sp_x, sp_y):
        input_dim = input.dim()
        assert input.dim() == 4
        assert input.size(0) == 1

        output = torch.zeros_like(input)
        v_max, c_max = torch.max(input, 1)
        c_max_flatten = c_max.view(-1)
        v_max_flatten = v_max.view(-1)
        ind = c_max_flatten
        return output, ind, v_max_flatten


class ShiftOpr(object):
    def buildAutoencoder(self, target_img, normalize, interpolate, nonmask_point_idx, mask_point_idx, patch_size=1,
                         stride=1):
        nDim = 3
        assert target_img.dim() == nDim
        C = target_img.size(0)
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.Tensor
        patches_all, patches_part, patches_mask = self._extract_patches(target_img, patch_size, stride,
                                                                        nonmask_point_idx, mask_point_idx)
        npatches_part = patches_part.size(0)
        npatches_all = patches_all.size(0)
        conv_enc_non_mask, conv_dec_non_mask = self._build(patch_size, stride, C, patches_part, npatches_part,
                                                           normalize, interpolate)
        conv_enc_all, conv_dec_all = self._build(patch_size, stride, C, patches_all, npatches_all, normalize,
                                                 interpolate)
        return conv_enc_all, conv_enc_non_mask, conv_dec_all, conv_dec_non_mask, patches_part, patches_mask

    def _build(self, patch_size, stride, C, target_patches, npatches, normalize, interpolate):
        enc_patches = target_patches.clone()
        for i in range(npatches):
            enc_patches[i] = enc_patches[i] * (1 / (enc_patches[i].norm(2) + 1e-8))
        conv_enc = nn.Conv2d(C, npatches, kernel_size=patch_size, stride=stride, bias=False)
        conv_enc.weight.data = enc_patches
        if normalize:
            raise NotImplementedError
        if interpolate:
            raise NotImplementedError
        conv_dec = nn.ConvTranspose2d(npatches, C, kernel_size=patch_size, stride=stride, bias=False)
        conv_dec.weight.data = target_patches
        return conv_enc, conv_dec

    def _extract_patches(self, img, patch_size, stride, nonmask_point_idx, mask_point_idx):
        n_dim = 3
        assert img.dim() == n_dim
        kH, kW = patch_size, patch_size
        dH, dW = stride, stride
        input_windows = img.unfold(1, kH, dH).unfold(2, kW, dW)
        i_1, i_2, i_3, i_4, i_5 = input_windows.size(0), input_windows.size(1), input_windows.size(
            2), input_windows.size(3), input_windows.size(4)
        input_windows = input_windows.permute(1, 2, 0, 3, 4).contiguous().view(i_2 * i_3, i_1, i_4, i_5)

        patches_all = input_windows
        patches = input_windows.index_select(0, nonmask_point_idx)
        maskpatches = input_windows.index_select(0, mask_point_idx)
        return patches_all, patches, maskpatches

    def _extract_patches_mask(self, img, patch_size, stride, nonmask_point_idx, mask_point_idx):
        n_dim = 3
        assert img.dim() == n_dim
        kH, kW = patch_size, patch_size
        dH, dW = stride, stride
        input_windows = img.unfold(1, kH, dH).unfold(2, kW, dW)
        i_1, i_2, i_3, i_4, i_5 = input_windows.size(0), input_windows.size(1), input_windows.size(
            2), input_windows.size(3), input_windows.size(4)
        input_windows = input_windows.permute(1, 2, 0, 3, 4).contiguous().view(i_2 * i_3, i_1, i_4, i_5)
        maskpatches = input_windows.index_select(0, mask_point_idx)
        return maskpatches