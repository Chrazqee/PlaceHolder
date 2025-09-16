# Codes are taken from BPNet, CVPR'21
# https://github.com/wbhu/BPNet/blob/main/dataset/voxelizer.py

import collections
import numpy as np
from voxelization_utils import sparse_quantize
from scipy.linalg import expm, norm


# Rotation matrix along axis with angle theta
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


class Voxelizer:

    def __init__(self,
                 voxel_size=1,
                 clip_bound=None,  # point map output by vggt should bound the points location! The argument should be useful!
                 use_augmentation=False,
                 scale_augmentation_bound=None,
                 rotation_augmentation_bound=None,
                 translation_augmentation_ratio_bound=None,
                 ignore_label=255):
        '''
        Args:
          - voxel_size: side length of a voxel

          - clip_bound: boundary of the voxelizer. Points outside the bound will be deleted
            expects either None or an array like ((-100, 100), (-100, 100), (-100, 100)).

          - scale_augmentation_bound: None or (0.9, 1.1)

          - rotation_augmentation_bound: None or ((np.pi / 6, np.pi / 6), None, None) for 3 axis.
            Use random order of x, y, z to prevent bias.

          - translation_augmentation_bound: ((-5, 5), (0, 0), (-10, 10))

          - ignore_label: label assigned for ignore (not a training label).
        '''
        self.voxel_size = voxel_size
        self.clip_bound = clip_bound
        self.ignore_label = ignore_label

        # Augmentation
        self.use_augmentation = use_augmentation
        self.scale_augmentation_bound = scale_augmentation_bound
        self.rotation_augmentation_bound = rotation_augmentation_bound
        self.translation_augmentation_ratio_bound = translation_augmentation_ratio_bound

    def get_transformation_matrix(self):
        """
        这里的 transformation 只进行 rigid_transformation 的 旋转 变换, 而不进行 平移变换
        """
        voxelization_matrix, rotation_matrix = np.eye(4), np.eye(4)
        # Get clip boundary from config or pointcloud.
        # Get inner clip bound to crop from.

        # Transform pointcloud coordinate to voxel coordinate.
        # 1. Random rotation
        rot_mat = np.eye(3)
        if self.use_augmentation and self.rotation_augmentation_bound is not None:
            if isinstance(self.rotation_augmentation_bound, collections.Iterable):
                rot_mats = []
                for axis_ind, rot_bound in enumerate(self.rotation_augmentation_bound):
                    theta = 0
                    axis = np.zeros(3)
                    axis[axis_ind] = 1
                    if rot_bound is not None:
                        theta = np.random.uniform(*rot_bound)
                    rot_mats.append(M(axis, theta))
                # Use random order
                np.random.shuffle(rot_mats)
                rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]
            else:
                raise ValueError()
        rotation_matrix[:3, :3] = rot_mat
        # 2. Scale and translate to the voxel space.
        scale = 1 / self.voxel_size
        if self.use_augmentation and self.scale_augmentation_bound is not None:
            scale *= np.random.uniform(*self.scale_augmentation_bound)
        np.fill_diagonal(voxelization_matrix[:3, :3], scale)
        # Get final transformation matrix.
        return voxelization_matrix, rotation_matrix

    def clip(self, coords, center=None, trans_aug_ratio=None):
        bound_min = np.min(coords, 0).astype(float)
        bound_max = np.max(coords, 0).astype(float)
        bound_size = bound_max - bound_min
        if center is None:
            center = bound_min + bound_size * 0.5
        lim = self.clip_bound
        if trans_aug_ratio is not None:
            trans = np.multiply(trans_aug_ratio, bound_size)
            center += trans
        # Clip points outside the limit
        clip_inds = ((coords[:, 0] >= (lim[0][0] + center[0])) &
                     (coords[:, 0] < (lim[0][1] + center[0])) &
                     (coords[:, 1] >= (lim[1][0] + center[1])) &
                     (coords[:, 1] < (lim[1][1] + center[1])) &
                     (coords[:, 2] >= (lim[2][0] + center[2])) &
                     (coords[:, 2] < (lim[2][1] + center[2])))
        return clip_inds

    def voxelize(self, coords, feats, labels, center=None, link=None, return_ind=False):
        """
        
        Returns:
            - coords: voxelized coordinates, int array of shape (N, 3)
            - feats: voxelized features, float array of shape (N, C).  点的特征, 如颜色(3)、法向量(3)等
            - labels: voxelized labels, int array of shape (N,)
            - inds_reconstruct: indices to reconstruct the original points from the voxelized points, int array of shape (M,). 每个原始点被映射到哪个体素索引
        """
        assert coords.shape[1] == 3 and coords.shape[0] == feats.shape[0] and coords.shape[0]
        if self.clip_bound is not None:
            trans_aug_ratio = np.zeros(3)
            if self.use_augmentation and self.translation_augmentation_ratio_bound is not None:
                for axis_ind, trans_ratio_bound in enumerate(self.translation_augmentation_ratio_bound):
                    trans_aug_ratio[axis_ind] = np.random.uniform(*trans_ratio_bound)

            clip_inds = self.clip(coords, center, trans_aug_ratio)
            if clip_inds.sum():
                coords, feats = coords[clip_inds], feats[clip_inds]
                if labels is not None:
                    labels = labels[clip_inds]

        # Get rotation and scale
        # s = 1 / self.voxel_size   # r -> random value
        # M_v : [[s, 0, 0, 0]       M_r:[[r11, r12, r13, 0]  
        #        [0, s, 0, 0]            [r21, r22, r23, 0] 
        #        [0, 0, s, 0]            [r31, r32, r33, 0] 
        #        [0, 0, 0, 1]]           [ 0,   0,   0,  1]]
        M_v, M_r = self.get_transformation_matrix()  # voxelization_matrix, rotation_matrix 一个是缩放矩阵，一个是旋转矩阵
        # Apply transformations
        rigid_transformation = M_v 
        if self.use_augmentation:
            rigid_transformation = M_r @ M_v  # 先拉伸, 再旋转

        # (n, 3) -> (n, 4) [x, y, z, 1] 的齐次坐标
        homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))

        # return value, 
        coords_aug = np.floor(homo_coords @ rigid_transformation.T[:, :3])  # (n, 4) @ (4, 3) => (n, 3)

        # Align all coordinates to the origin.
        min_coords = coords_aug.min(0)
        # M_t = np.eye(4)
        # M_t[:3, -1] = -min_coords
        # rigid_transformation = M_t @ rigid_transformation

        coords_aug = np.floor(coords_aug - min_coords)

        # 将多个落在同一个体素格子里的点合并为一个，实现稀疏体素化
        # inds.shape != inds_reconstruct.shape
        # inds_reconstruct == inds_reverse
        inds, inds_reconstruct = sparse_quantize(coords_aug, return_index=True)  # [ ]: how inds(index) work!

        coords_aug, feats, labels = coords_aug[inds], feats[inds], labels[inds]

        # Normal rotation  法线旋转
        if feats.shape[1] > 6:
            feats[:, 3:6] = feats[:, 3:6] @ (M_r[:3, :3].T)

        if return_ind:
            return coords_aug, feats, labels, np.array(inds_reconstruct), inds
        if link is not None:
            return coords_aug, feats, labels, np.array(inds_reconstruct), link[inds]
        
        return coords_aug, feats, labels, np.array(inds_reconstruct)


if __name__ == "__main__":
    # [x]: check function of this class and how to use!
    # - step1: initialize the Voxelizer
    voxelizer = Voxelizer(voxel_size=0.05,
                          clip_bound=None,
                          use_augmentation=False,
                          scale_augmentation_bound=None,
                          rotation_augmentation_bound=None,
                          translation_augmentation_ratio_bound=None,
                          )
    # - step2: prepare input data
    coords = np.random.rand(100, 3)                 # 点的坐标

    print(f"coords first 5 lines: \n {coords[:5]}") 

    feats = np.random.rand(100, 6)                  # [x]: check
    labels = np.random.randint(0, 10, size=(100,))  # 普通的标签值 # [ ]: 思考🤔: 无监督时无标签怎么办! 用 0 填充, 作为 identity 吗
    # - step3: voxelize the input data
    voxelized_data = voxelizer.voxelize(coords, feats, labels)
    coords_aug, feats, labels, inds_reconstruct = voxelized_data
    # [x]: check the output data
    print(f"coords_aug size: {coords_aug.shape}, coords_aug first 5 lines: \n {coords_aug[:5]}")                            # 对应点落到体素中的体素坐标
    print(f"feats size: {feats.shape}, feats first 5 lines: \n {feats[:5]}")                                                # 
    print(f"labels size: {labels.shape}, labels first 5 lines: \n {labels[:5]}")                                            # 
    print(f"inds_reconstruct size: {inds_reconstruct.shape}, inds_reconstruct first 5 lines: \n {inds_reconstruct[:5]}")    # 
