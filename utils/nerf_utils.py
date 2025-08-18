# ======================================================
# source code from: https://github.com/zyp123494/DynaVol
# ======================================================
import torch
import numpy as np
from einops import rearrange


def get_rays(H, W, K, c2w, inverse_y=False, flip_x=False, flip_y=False, mode='center'):
    """
    K: 内参矩阵
    c2w: 相机坐标转换到世界坐标, 外参矩阵
    returns:
        rays_o: 射线起点
        rays_d: 射线方向
    """
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device),
        indexing='ij'
        )  # pytorch's meshgrid has indexing='ij'

    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i + 0.5, j + 0.5
    elif mode == 'random':
        i = i + torch.rand_like(i)
        j = j + torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip(dims=(1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        # 针孔摄像机反投影公式, 获得每个像素点的射线方向
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], dim=-1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], dim=-1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # c2w[:3, 3] -> [t_x, t_y, t_z]
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d

def get_rays_of_a_view(H, W, K, c2w, ndc, mode='center'):
    """
    returns: 射线起点, 射线方向, 单位化的射线方向
    """
    rays_o, rays_d = get_rays(H, W, K, c2w, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)  # 射线方向的单位化
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs

def sin_emb(input_data, n_freq):  # n_freq 的作用是，学习更多高频的信息，多频正余弦编码
    """
    n_freq: torch.FloatTensor([(2**i) for i in range(train_cfg.n_freq)]).to(input_data.device)
        train_cfg.n_freq=5
    return: embedding
    """
    # n_freq = torch.FloatTensor([(2**i) for i in range(n_freq)]).to(input_data.device)
    # input_data: (4096, 3); n_freq: (4); (input_data.unsqueeze(-1): (4096, 3, 1) * n_freq).shape: (4096, 3, 4)
    input_data_emb = (input_data.unsqueeze(-1) * n_freq).flatten(-2)  # (4096, 12)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin, input_data_cos], -1)  # (N, 36)
    return input_data_emb


def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')


def compute_3d_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(  
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o + rays_d * near, rays_o + rays_d * far])
        else:
            pts_nf = torch.stack([rays_o + viewdirs * near, rays_o + viewdirs * far])  # 视锥体 -> [近截面, 远截面]

        xyz_min = torch.minimum(xyz_min, pts_nf.amin(dim=(0,1,2)))  # pts_nf.amin(dim=(0,1,2)) 所有轴中最小的那一个
        xyz_max = torch.maximum(xyz_max, pts_nf.amax(dim=(0,1,2)))  # 最大的那一个
    print('compute_bbox_by_cam_frustrm -> xyz_min: ', xyz_min)
    print('compute_bbox_by_cam_frustrm -> xyz_max: ', xyz_max)
    print('compute_bbox_by_cam_frustrm: finished')
    return xyz_min, xyz_max
