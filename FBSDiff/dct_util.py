import os
import numpy as np
import torch
from PIL import Image

from FBSDiff.tools import create_model, load_state_dict


def img_resize(img_folder, resize_h, resize_w):
    for img_path in os.listdir(img_folder):
        Image.open(os.path.join(img_folder, img_path)).resize((resize_h, resize_w)). \
            save(os.path.join(img_folder, img_path))


def dct(x, norm=None):
    '''
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    '''
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))
    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
    V = 2 * V.view(*x_shape)
    return V


def idct(X, norm=None):
    '''
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    '''
    x_shape = X.shape
    N = x_shape[-1]
    X_v = X.contiguous().view(-1, x_shape[-1]) / 2
    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2
    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)
    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r
    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]
    return x.view(*x_shape)


def dct_2d(x, norm=None):
    '''
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT_II of the signal over the last 2 dimensions
    '''
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    '''
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimension
    '''
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    '''
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT_II of the signal over the last 3 dimensions
    '''
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    '''
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_3d(dct_3d(x)) == x
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimension
    '''
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


def low_pass(dct, threshold):
    '''
    dct: tensor of ... h, w shape
    threshold: integer number above which to zero out
    '''
    h, w = dct.shape[-2], dct.shape[-1]
    assert 0 <= threshold <= h + w - 2, 'invalid value of threshold'
    vertical = torch.range(0, h-1)[..., None].repeat(1, w).cuda()
    horizontal = torch.range(0, w-1)[None, ...].repeat(h, 1).cuda()
    mask = vertical + horizontal
    while len(mask.shape) != len(dct.shape):
        mask = mask[None, ...]
    dct = torch.where(mask > threshold, torch.zeros_like(dct), dct)
    return dct

def low_pass_and_shuffle(dct, threshold):
    '''
    dct: tensor of ... h, w shape
    threshold: integer number above which to zero out
    '''
    h, w = dct.shape[-2], dct.shape[-1]
    assert 0 <= threshold <= h + w - 2, 'invalid value of threshold'
    vertical = torch.range(0, h-1)[..., None].repeat(1, w).cuda()
    horizontal = torch.range(0, w-1)[None, ...].repeat(h, 1).cuda()
    mask = vertical + horizontal
    while len(mask.shape) != len(dct.shape):
        mask = mask[None, ...]
    dct = torch.where(mask > threshold, torch.zeros_like(dct), dct)
    for i in range(0, threshold + 1):         # 1 ~ threshold
        dct = shuffle_one_frequency(i, dct)
    return dct


def shuffle_one_frequency(n, dct_tensor):
    h_num = torch.arange(n + 1)
    h_num = h_num[torch.randperm(n + 1)]
    v_num = n - h_num
    dct_tensor_copy = dct_tensor.clone()
    for i in range(n + 1):  # 0 ~ n
        dct_tensor[:, :, i, n - i] = dct_tensor_copy[:, :, v_num[i], h_num[i]]
    return dct_tensor


def high_pass(dct, threshold):
    '''
    dct: tensor of ... h, w shape
    threshold: integer number below which to zero out
    '''
    h, w = dct.shape[-2], dct.shape[-1]
    assert 0 <= threshold <= h + w - 2, 'invalid value of threshold'
    vertical = torch.range(0, h-1)[..., None].repeat(1, w).cuda()
    horizontal = torch.range(0, w-1)[None, ...].repeat(h, 1).cuda()
    mask = vertical + horizontal
    while len(mask.shape) != len(dct.shape):
        mask = mask[None, ...]
    dct = torch.where(mask < threshold, torch.zeros_like(dct), dct)
    return dct


def dct_swap(dct1, dct2, k):
    # dct1: c, h, w
    # dct2: c, h, w
    dct2_map = torch.flatten(torch.sum(torch.abs(dct2), dim=0))  # h * w
    sorted, index = torch.sort(dct2_map, descending=True)
    c, h, w = dct1.shape
    dct1 = dct1.reshape(c, -1).permute(1, 0)   # hw, c
    dct2 = dct2.reshape(c, -1).permute(1, 0)   # hw, c

    dct1[index[:k], ...] = dct2[index[:k], ...]

    dct1 = dct1.permute(1, 0).reshape(c, h, w)  # c, h, w

    return dct1


def dct_align(dct1, dct2):
    # dct1: n, c, h, w
    # dct2: n, c, h, w
    n, c, h, w = dct1.shape

    dct1_map = torch.flatten(torch.sum(dct1, dim=1), start_dim=1)  # n, h*w
    dct2_map = torch.flatten(torch.sum(dct2, dim=1), start_dim=1)  # n, h*w

    sorted1, index1 = torch.sort(dct1_map, descending=True, dim=-1)
    sorted2, index2 = torch.sort(dct2_map, descending=True, dim=-1)

    dct1 = dct1.reshape(n*c, -1)  # nc, hw
    dct2 = dct2.reshape(n*c, -1)  # nc, hw

    dct1[..., index1] = dct2[..., index2]
    dct1 = dct1.reshape(n, c, h, w)  # n, c, h, w

    return dct1


def dct_separate_align(feat_ref, feat_src):
    # feat_ref: n, c, h, w
    # feat_src: n, c, h, w
    n, c, h, w = feat_ref.shape
    dct_ref = dct_2d(feat_ref, norm='ortho')  # n, c, h, w
    dct_src = dct_2d(feat_src, norm='ortho')  # n, c, h, w
    dct_channels = []
    for i in range(c):
        dct_channel_ref = dct_ref[:, i, ...].reshape(n, -1)  # n, hw
        dct_channel_src = dct_src[:, i, ...].reshape(n, -1)  # n, hw

        _, index_ref = torch.sort(torch.abs(dct_channel_ref), descending=True, dim=-1)  # n, hw
        _, index_src = torch.sort(torch.abs(dct_channel_src), descending=True, dim=-1)  # n, hw

        gathered_src_values = torch.gather(input=dct_channel_src, index=index_src, dim=-1)
        dct_channel_ref = dct_channel_ref.scatter(src=gathered_src_values, index=index_ref, dim=-1)

        dct_channel_ref = dct_channel_ref.reshape(n, 1, h, w)  # n, 1, h, w
        dct_channels.append(dct_channel_ref)
    dct_channels = torch.cat(dct_channels, dim=1)  # n, c, h, w
    aligned = idct_2d(dct_channels, norm='ortho')
    return aligned


def dct_separate_align2(feat_ref, feat_src, topk):
    # feat1: n, c, h, w
    # feat2: n, c, h, w
    n, c, h, w = feat_ref.shape
    alligned_channels = []

    for i in range(c):
        feat_ref_channel = feat_ref[:, i, ...]  # n, h, w
        feat_src_channel = feat_src[:, i, ...]  # n, h, w

        feat_ref_dct = dct_2d(feat_ref_channel, norm='ortho').reshape(n, -1)  # n, hw
        feat_src_dct = dct_2d(feat_src_channel, norm='ortho').reshape(n, -1)  # n, hw

        _, index_ref = torch.sort(feat_ref_dct, descending=True, dim=-1)  # n, hw
        _, index_src = torch.sort(feat_src_dct, descending=True, dim=-1)  # n, hw

        values_src_index = torch.gather(input=feat_src_dct, index=index_src[:, :topk], dim=-1)
        values_ref_index = torch.gather(input=feat_src_dct, index=index_ref[:, :topk], dim=-1)
        feat_src_dct.scatter_(src=values_ref_index, index=index_src[:, :topk], dim=-1)
        feat_src_dct.scatter_(src=values_src_index, index=index_ref[:, :topk], dim=-1)

        values_src_index = torch.gather(input=feat_src_dct, index=index_src[:, -topk:], dim=-1)
        values_ref_index = torch.gather(input=feat_src_dct, index=index_ref[:, -topk:], dim=-1)
        feat_src_dct.scatter_(src=values_ref_index, index=index_src[:, -topk:], dim=-1)
        feat_src_dct.scatter_(src=values_src_index, index=index_ref[:, -topk:], dim=-1)

        aligned_channel = idct_2d(feat_src_dct.reshape(n, 1, h, w), norm='ortho')  # n, 1, h, w
        alligned_channels.append(aligned_channel)

    aligned = torch.cat(alligned_channels, dim=1)  # n, c, h, w
    return aligned


def spatial_align(feat_ref, feat_src):
    # feat_ref: n, c, h, w
    # feat_src: n, c, h, w
    n, c, h, w = feat_ref.shape
    channel_list = []
    for i in range(c):
        ref = feat_ref[:, i, ...]  # n, h, w
        src = feat_src[:, i, ...]  # n, h, w
        ref = ref.reshape(n, -1)
        src = src.reshape(n, -1)
        _, ref_index = torch.sort(ref, descending=True, dim=-1)
        _, src_index = torch.sort(src, descending=True, dim=-1)

        values_ref_index = torch.gather(input=src, index=src_index, dim=-1)
        ref.scatter_(src=values_ref_index, index=ref_index, dim=-1)

        channel_list.append(ref.reshape(n, 1, h, w))
    aligned = torch.cat(channel_list, dim=1)  # n, c, h, w
    return aligned



def dct_group_align(dct1, dct2):
    # dct1: c, h, w
    # dct2: c, h, w

    c, h, w = dct1.shape

    assert h == w, 'the width and height should be the same'

    dct2_map = torch.sum(dct2, dim=0)   # h, w
    dct1_map = torch.sum(dct1, dim=0)   # h, w

    total_sum = h + w - 2
    dct1_map_list = []
    dct2_map_list = []
    dct1_list = []
    dct2_list = []

    for sum in range(total_sum + 1):    # 0 ~ total_sum
        if sum <= h-1:
            for row in range(sum + 1):  # 0 ~ sum
                col = sum - row         # 0 ~ sum
                dct1_map_list.append(dct1_map[row, col])
                dct2_map_list.append(dct2_map[row, col])
                dct1_list.append(dct1[..., row, col])
                dct2_list.append(dct2[..., row, col])
            sorted1, index1 = torch.sort(torch.from_numpy(np.array(dct1_map_list)), descending=True)
            sorted2, index2 = torch.sort(torch.from_numpy(np.array(dct2_map_list)), descending=True)
            dct1_list_tensor = torch.stack(dct1_list, dim=0)  # n, c
            dct2_list_tensor = torch.stack(dct2_list, dim=0)  # n, c
            dct1_list_tensor[index1] = dct2_list_tensor[index2]
            for row in range(sum + 1):
                col = sum - row
                dct1[..., row, col] = dct1_list_tensor[0]
                dct1_list_tensor = dct1_list_tensor[1:]
            print(str(sum) + ' finished')

        else:  # h <= sum <= h + w - 2
            for row in range(sum - (h-1), h):
                col = sum - row
                dct1_map_list.append(dct1_map[row, col])
                dct2_map_list.append(dct2_map[row, col])
                dct1_list.append(dct1[..., row, col])
                dct2_list.append(dct2[..., row, col])
            sorted1, index1 = torch.sort(torch.from_numpy(np.array(dct1_map_list)), descending=True)
            sorted2, index2 = torch.sort(torch.from_numpy(np.array(dct2_map_list)), descending=True)
            dct1_list_tensor = torch.stack(dct1_list, dim=0)  # n, c
            dct2_list_tensor = torch.stack(dct2_list, dim=0)  # n, c
            dct1_list_tensor[index1] = dct2_list_tensor[index2]
            for row in range(sum - (h - 1), h):
                col = sum - row
                dct1[..., row, col] = dct1_list_tensor[0]
                dct1_list_tensor = dct1_list_tensor[1:]
            print(str(sum) + ' finished')

        dct1_map_list.clear()
        dct2_map_list.clear()
        dct1_list.clear()
        dct2_list.clear()

    return dct1










