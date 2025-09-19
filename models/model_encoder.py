import torch
from torchvision import transforms
import cv2
import torch.nn as nn
from torch_scatter import scatter, scatter_add, scatter_mean


try:
    from .mink_unet import mink_unet as MinkUnet
except ImportError as e:
    print("[models] Relative import failed:", e)
    try:
        from mink_unet import mink_unet as MinkUnet
    except ImportError as e2:
        print("[models] Absolute import also failed:", e2)
        raise e2
    
from hydra.utils import instantiate


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

import sys
sys.path.append("/home/hanqi/codes/PlaceHolder")


class VisualEncoder2D(nn.Module):
    """ A class to load different pretrained models based on the model name.

    Currently supports:
        - "dinov2-vitb14": DINOv2 ViT-B/14
        - "dinov3-vitb16": DINOv3 ViT-B/16
    
    """
    def __init__(self, 
                 model_name,
                 image_size=518,
                 keep_attn=False,
                 ):
        """ Initialize the model loader with the model name.

        Args:
            model_name (str): Name of the model to load.
            keep_attn (bool): Whether to keep attention weights (if supported by the model).
        """
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size
        self.keep_attn = keep_attn
        self.load_model()


    def load_model(self):
        if self.model_name == "dinov3-vitb16":
            self._load_dinov3_vitb16()
            self.patch_size = 16
        elif self.model_name == "dinov2-vitb14":
            self._load_dinov2_vitb14()
            self.patch_size = 14
        else:
            raise ValueError(f"Model {self.model_name} not supported yet.")
        
        # Freeze the model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        return self.model


    def _load_dinov3_vitb16(self):
        # Using transformers library to load DINOv3 ViT-B/16
        from transformers import AutoModel, AutoConfig

        pretrained_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        config = AutoConfig.from_pretrained(pretrained_model_name)

        if self.keep_attn:
            config.attn_implementation = "eager"
            config.output_attentions = True

        self.model = AutoModel.from_pretrained(
            pretrained_model_name,
            config=config, 
            device_map="auto",  # Automatically place model on available devices(multi-gpu if available)
        )
    

    def _load_dinov2_vitb14(self):
        # wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")


    def forward(self, x):
        """
        Forward pass through the loaded model.
        Different models have different forward methods. Handle them accordingly.
        """
        if self.model_name == "dinov3-vitb16":
            return self._forward_dinov3_vitb16(x)
        elif self.model_name == "dinov2-vitb14":
            return self._forward_dinov2_vitb14(x)
        else:
            raise ValueError("Model not loaded. Call load_model() first.")


    # [x]: 检查是否 work => last_hidden_state 包含全局的语义 cls token，slot attention 只需要局部的 patch tokens 即可
    def _forward_dinov3_vitb16(self, x):
        """
        Args:
            - x: Input tensor of shape (B, 3, H, W)

        Returns:
            - patch tokens of shape (B, tokens, 768)

        outputs includes:
            - last_hidden_state: (B, tokens+1+4, 768)
            - attentions: (B, num_heads, tokens+1+4, tokens+1+4) if keep_attn is True
            - pooler_output: (B, 768)
        """
        outputs = self.model(x)
        last_hidden_state = outputs.last_hidden_state  # (B, token, 768)
        return last_hidden_state[:, 5:, :]  # Remove CLS token and 4 register tokens!
    

    def _forward_dinov2_vitb14(self, x):
        # :arg x: (B, 3, H, W)
        # :return x: (B, token, 768)
        x = self.model.prepare_tokens_with_masks(x)
        for blk in self.model.blocks:
            x = blk(x)
        x = x[:, 1:]  # Remove CLS token!
        return x


class VisualEncoder3D(nn.Module):
    def __init__(self, 
                 model_name='MinkUnet',
                 coords_dim=3,
                 feats_dim=3,
                 last_dim=384,
                 arch_3d='MinkUNet34C',
        ):
        """
        Construct a 3D visual encoder using MinkowskiUNet.
        Args:
            - model_name(str): Definiate the model name
            - feats_dim(int): The dimension of features of every points, aka. feats.shape[1]
            - last_dim (int): The output feature dimension of the 3D encoder.
            - arch_3d (str): The architecture of the 3D model. Default is 'MinkUNet18A'.

        """
        super().__init__()
        self.model_name = model_name
        self.coords_dim = coords_dim
        self.feats_dim = feats_dim
        self.last_dim = last_dim
        self.arch_3d = arch_3d
        net3d = self._constructor3d()  # MinkowskiNet for 3D point clouds
        self.net3d = net3d


    def forward(self, x):
        """
        x is a SparseTensor, typically organized as: SparseTensor(features, coordinates), coordinates is (N, 4) with first column being batch indices.
        """
        return self.net3d(x)


    def _constructor3d(self):
        """For extensible!"""
        if self.model_name == 'MinkUnet':
            model = MinkUnet(
                in_channels=self.feats_dim, 
                out_channels=self.last_dim, 
                D=self.coords_dim,          # 表示处理 3D 点云或体素数据, 坐标是 [x, y, z]
                arch=self.arch_3d)
        else:
            raise NotImplementedError(f"The model {self.model_name} was not implemented.")
        return model


class Aggregator(nn.Module):
    _AGGREGATION_FN = {"cross_attn", }
    def __init__(self, 
                 visual_encoder_2d,
                 visual_encoder_3d,
                 mlp, 
                 aggregator=None,               # 在 yaml 中配置 aggregation 参数
                 aggregation_fn="cross_attn",
                 aggregation_threshold=0.6,      # For look for index, which is most similar to the query(aka. 2D tokens)
                 aggregation_learnable=True
                 ):
        super().__init__()
        assert aggregation_fn in Aggregator._AGGREGATION_FN, f"You must point out the aggregated function in the set: {Aggregator._AGGREGATION_FN}."
        
        self.visual_model_2d = instantiate(visual_encoder_2d, _recursive_=False)
        self.visual_model_3d = instantiate(visual_encoder_3d, _recursive_=False)
        self.mlp = mlp
        # aggregation_fn and aggregator should be consistent, the class should be undertake the function
        self.aggregator = aggregator
        self.aggregation_fn = aggregation_fn
        self.aggregation_threshold = aggregation_threshold if aggregation_fn == "cross_attn" else None
        if aggregation_learnable:
            visual_encoder_2d_patch_size = self.visual_model_2d.patch_size
            # 现在需要知道 2d 的 images 的尺寸
            visual_encoder_2d_image_size = self.visual_model_2d.image_size
            assert visual_encoder_2d_image_size % visual_encoder_2d_patch_size == 0, "Image size must be divisible by patch size."
            num_patch_per_side = visual_encoder_2d_image_size // visual_encoder_2d_patch_size
            self.aggregation_threshold = nn.Parameter(
                torch.tensor(aggregation_threshold * torch.ones(num_patch_per_side, device=self.device)))  # [ ]: 检查 device 设置是否有误

        self._initialize_aggregation()
        

    def forward(self, x_2d, x_3d):
        """
        Args:
            - x_2d: (B, 3, H, W)
            - x_3d: SparseTensor, with coordinates (N, C) and features (N, F)
        """
        feats_2d = self.visual_model_2d(x_2d)

        # Note: feats_3d should not feed in `_aggregate_features` directly, because it contained both point features and coordinate features.
        feats_3d = self.visual_model_3d(x_3d)

        feats_3d_F = feats_3d.F  # (N, F)
        feats_3d_C = feats_3d.C  # (N, 4), with first column being batch indices.
        # 把 batch cat 到 output_feature 的第一列, 用来区分不同的 batch 
        _feats_3d_F = torch.cat([feats_3d_C[:, :1].float(), feats_3d_F], dim=-1)  # (N, F+1)

        feats = self._aggregate_features(feats_2d, _feats_3d_F)

        return feats

    def _initialize_aggregation(self):
        # We can also add some other function when some functions need to be instantiated.
        # To be elegant!
        self.feature_aggregator = instantiate(self.aggregator, _recursive_=False)  # [ ]: 一层 cross_attention layer 够用吗?
        self.mlp = instantiate(self.mlp, _recursive_=False)
        # instantiate the alignment module and the pooling module.


    # [ ]: Implement this function
    def _aggregate_features(self, feats_2d, feats_3d):
        """
        Args:
            - feats_2d: features from DINO, with high-level semantic information
            - feats_3d: features from Minkowski unet, with high-level geometry information
        """
        if self.aggregation_fn == "cross_attn":
            # 这里只需要把 feats_3d 送入 MLP 即可, 得到的输出的每个点的特征的维度与 token 的维度相同
            feats_3d = self.mlp(feats_3d)
            
            # Cross Attention aggregation
            if self.feature_aggregator is not None:
                # outputs 是 query 到的结果, 作为新的 token 
                outputs, attn_maps = self.feature_aggregator(feats_3d, feats_2d, None)  # cross_attention 的输出结果
                index = self._find_most_similar_indices(attn_map=attn_maps)             # index: list(tensor()) \in [B, n_tokens, n_points]
                feats = self._do_aggregate(outputs, feats_3d, index)
            else:
                raise ValueError("The instance of feature_aggregator is None, please check your config file.")

        elif self.aggregation_fn == "xxx":
            raise NotImplementedError("cross_attn not implemented yet.")
        else:
            raise ValueError(f"Aggregation function {self.aggregation_fn} not supported yet.")

        return feats


    def _do_aggregate(self, features, feats_3d, index):
        # select points according to index and merge the poins' feature, we can adopt `mean` or `sum`, etc.
        # 这里好像可以通过使用 sactter 函数来实现
        # 把 feats_3d 中跟 index 对应的 features merge 起来, 然后返回, 这是最后一步了
        # [ ]: 这里要看 feats_2d 的 batch_size
        B, N_t, C = features.shape  # B: batch size, N_t: number of tokens, C: feature dimension
        for b in range(B):
            idx = index[b]
            feats_3d_batch = feats_3d[b: b + 1, ...]  # (1, N_points, C)
            # index 是一个 vector, 里面是 0/1, 表示哪些 points 被选中
            selected_feats = feats_3d_batch[:, idx, :]  # (1, N_selected, C)
            # 对于被选中的 features 进行聚合
            aggregated_feat = scatter_mean(selected_feats, torch.zeros(selected_feats.shape[1], device=selected_feats.device).long(), dim=1)  # (1, 1, C)
        pass


    def _find_most_similar_indices(self, attn_map):
        """
        Args:
            - attn_map: (B, n_queries, n_kv), attention map from cross attention layer
            - threshold: float, threshold to filter the most similar indices
        Returns:
            - indices: list of list, each inner list contains the indices of the most similar kv for each query
        """
        if attn_map.dim() != 3:
            n_queries, n_kv = attn_map.shape
            B = 1
        else:
            B, n_queries, n_kv = attn_map.shape
        points_to_tokens_indices = []
        for b in range(B):
            points_to_tokens_scores = attn_map[b, :]  # matrix
            points_to_tokens_scores /= points_to_tokens_scores.norm(dim=0)
            index = points_to_tokens_scores >= self.aggregation_threshold  # True or False 矩阵, 对应位置表示是否匹配上了
            points_to_tokens_indices.append(index)

        # [ ]: tensor 矩阵变换, 是不是应该把这个 tensor stack 起来
        return points_to_tokens_indices
    

if __name__ == "__main__":
    def testing():
        # [x]: test VisualEncoder2D
        import cv2
        import torchvision.transforms as transforms


        model_2d = VisualEncoder2D("dinov3-vitb16").eval()

        image_transform = transforms.Compose([
                transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                transforms.Resize((518, 518)),  # resize to 518x518
                transforms.Normalize(mean=_RESNET_MEAN, 
                                    std=_RESNET_STD),
            ])

        # we have some pices of images
        images_path_list = ["test/figs.assets/left113.png", 
                            "test/figs.assets/left118.png",
                            "test/figs.assets/left123.png",
                            "test/figs.assets/left128.png",
                            "test/figs.assets/left133.png",
                            "test/figs.assets/left138.png",
                            "test/figs.assets/left143.png",
                            "test/figs.assets/left148.png",
                    ] 
        image_tensor_list = []
        # read the image
        for i, image_path in enumerate(images_path_list):
            image = cv2.imread(image_path)  # BGR format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = image_transform(image)
            if i == 0:
                print(f"Image shape (C, H, W): {image_tensor.shape}")  # [3, H, W]
            image_tensor_list.append(image_tensor)

        # 将 image_tensor_list 中的 image_tensor stack 起来
        image_batch = torch.stack(image_tensor_list)  # [len(images_path_list), 3, H, W]

        # feed into our 2d model
        with torch.no_grad():
            outputs = model_2d(image_batch)
        print(outputs.shape)  # [2, 8045, 768] for dinov3-vitb16

    def testing_with_hydra():
        # [x]: test VisualEncoder2D with hydra
        from omegaconf import DictConfig, OmegaConf
        from hydra.utils import instantiate
        from loguru import logger

        # 我要显式指定 cfg
        cfg = OmegaConf.load("config/default.yaml")

        logger.info(cfg)

        model_2d = instantiate(cfg.Aggregator.visual_encoder_2d, _recursive_=False).eval()

        # 定义 transform
        image_transform = transforms.Compose([
                transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                transforms.Resize((518, 518)),  # resize to 518x518
                transforms.Normalize(mean=_RESNET_MEAN, 
                                     std=_RESNET_STD),
            ])
        # we have some pices of images
        images_path_list = ["test/figs.assets/left113.png",
                            "test/figs.assets/left118.png",
                            "test/figs.assets/left123.png",
                            "test/figs.assets/left128.png",
                            "test/figs.assets/left133.png",
                            "test/figs.assets/left138.png",
                            "test/figs.assets/left143.png",
                            "test/figs.assets/left148.png",
                    ]
        image_tensor_list = []
        # read the image
        for i, image_path in enumerate(images_path_list):
            image = cv2.imread(image_path)  # BGR format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = image_transform(image)
            if i == 0:
                print(f"Image shape (C, H, W): {image_tensor.shape}")  # [3, H, W]
            image_tensor_list.append(image_tensor)
        # 将 image_tensor_list 中的 image_tensor stack 起来
        image_batch = torch.stack(image_tensor_list)  # [len(images_path_list), 3, H, W]
        # feed into our 2d model
        with torch.no_grad():
            outputs = model_2d(image_batch)
        print(outputs.shape)  # [2, 8045, 768] for dinov3-vitb16
    
    def testing_3d():
        from MinkowskiEngine import SparseTensor
        # 1. 点 + 特征
        coords = torch.randint(0, 100, (300000, 3)).int()
        coords = torch.cat([torch.ones(300000, 1).int(), coords], dim=1)  # (N, 4), 第一列是 batch idx, 设置成 1
        feats = torch.randn(300000, 6)  # 假设每个点有6个特征, 例如 RGB + 法线
        # 2. 构造 SparseTensor
        st = SparseTensor(feats.cuda(non_blocking=True), 
                          coords.cuda(non_blocking=True))
        # 3. 构造模型
        # class VisualEncoder3D(nn.Module):
        #     def __init__(self, 
        #                  model_name='MinkUnet',
        #                  coords_dim=3,
        #                  feats_dim=3,
        #                  last_dim=384,
        #                  arch_3d='MinkUNet34C',
        #         ):
        model_3d = VisualEncoder3D(model_name='MinkUnet', 
                                   coords_dim=3,
                                   feats_dim=feats.shape[1], 
                                   last_dim=384, 
                                   arch_3d='MinkUNet18A').eval()
        model_3d = model_3d.cuda()
        # 4. 前向传播
        with torch.no_grad():
            out = model_3d(st)
        print(out.shape)  # (N, 384) 输出每个点的 384 维特征; 所以 MinkowskiNet 还是逐点运算的; # [x]: (CrossAttn)点的特征应该如何和 DINOv3 的 patch 特征对齐?
        pass

    def testing_3d_with_hydra():
        from omegaconf import DictConfig, OmegaConf
        from hydra.utils import instantiate
        cfg = OmegaConf.load("config/default.yaml")

        print(cfg.Aggregator.visual_encoder_3d)

        model_3d = instantiate(cfg.Aggregator.visual_encoder_3d, _recursive_=False).eval()

        print(model_3d)

        # [ ]: test CrossAttn useful or not
    # testing_3d_with_hydra()

    testing_3d()


