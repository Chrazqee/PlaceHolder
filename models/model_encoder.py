import torch
from torchvision import transforms
import cv2
import torch.nn as nn

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
                 keep_attn=False,
                 ):
        """ Initialize the model loader with the model name.

        Args:
            model_name (str): Name of the model to load.
            keep_attn (bool): Whether to keep attention weights (if supported by the model).
        """
        super().__init__()
        self.model_name = model_name
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
        x is a SparseTensor from MinkowskiEngine
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
                 aggregator=None,  # 在 yaml 中配置 aggregation 参数
                 aggregation_fn="cross_attn",
                 ):
        super().__init__()
        assert aggregation_fn in Aggregator._AGGREGATION_FN, f"You must point out the aggregated function in the set: {Aggregator._AGGREGATION_FN}."
        
        self.visual_model_2d = instantiate(visual_encoder_2d, _recursive_=False)
        self.visual_model_3d = instantiate(visual_encoder_3d, _recursive_=False)
        self.mlp = mlp
        # aggregation_fn and aggregator should be consistent, the class should be undertake the function
        self.aggregation_fn = aggregation_fn
        self.aggregator = aggregator

        self._initialize_aggregation()
        

    def forward(self, x_2d, x_3d):
        """
        Args:
            - x_2d: (B, 3, H, W)
            - x_3d: SparseTensor, with coordinates (N, C) and features (N, F)
        """
        feats_2d = self.visual_model_2d(x_2d)
        feats_3d = self.visual_model_3d(x_3d)

        feats = self._aggregate_features(feats_2d, feats_3d)

        return feats

    def _initialize_aggregation(self):
        # We can also add some other function when some functions need to be instantiated.
        # To be elegant!
        self.feature_aggregator = instantiate(self.aggregator, _recursive_=False)


        self.mlp = instantiate(self.mlp, _recursive_=False)


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
                self.feature_aggregator()
        



        elif self.aggregation_fn == "cross_attn":
            raise NotImplementedError("cross_attn not implemented yet.")
        else:
            raise ValueError(f"Aggregation function {self.aggregation_fn} not supported yet.")


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
        model_3d = VisualEncoder3D(feats_channels=feats.shape[1], last_dim=128, arch_3d='MinkUNet18A').eval()
        model_3d = model_3d.cuda()
        # 4. 前向传播
        with torch.no_grad():
            out = model_3d(st)
        print(out.shape)  # (N, 128) 输出每个点的128维特征; 所以 MinkowskiNet 还是逐点运算的; # [x]: (CrossAttn)点的特征应该如何和 DINOv3 的 patch 特征对齐?
        
    def testing_3d_with_hydra():
        from omegaconf import DictConfig, OmegaConf
        from hydra.utils import instantiate
        cfg = OmegaConf.load("config/default.yaml")

        print(cfg.Aggregator.visual_encoder_3d)

        model_3d = instantiate(cfg.Aggregator.visual_encoder_3d, _recursive_=False).eval()

        print(model_3d)

        # [ ]: test CrossAttn useful or not
    # testing_3d_with_hydra()
