import torch
import numpy as np


def get_rays(H, W, K, c2w, inverse_y=False, flip_x=False ,flip_y=False, mode='center'):

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
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], dim=-1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], dim=-1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # c2w[:3, 3] -> [t_x, t_y, t_z]
    return rays_o, rays_d


