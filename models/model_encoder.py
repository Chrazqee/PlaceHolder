import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn as nn


class VisualEncoder2D(nn.Module):
    """ A class to load different pretrained models based on the model name.

    Currently supports:
        - "dinov2-vitb14": DINOv2 ViT-B/14
        - "dinov3-vitb16": DINOv3 ViT-B/16
    
    """
    def __init__(self, model_name):
        """ Initialize the model loader with the model name.

        Args:
            model_name (str): Name of the model to load.
        """
        self.model_name = model_name


    def load_model(self):
        if self.model_name == "dinov3-vitb16":
            self._load_dinov3_vitb16()
        elif self.model_name == "dinov2-vitb14":
            self._load_dinov2_vitb14()
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
        config.attn_implementation = "eager"
        config.output_attentions = True

        self.model = AutoModel.from_pretrained(
            pretrained_model_name,
            config=config,
            device_map="auto",
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

    # [ ]: 检查是否 work
    def _forward_dinov3_vitb16(self, x):
        # :arg x: (B, 3, H, W)
        # :return x: (B, token, 768)
        outputs = self.model(x)
        last_hidden_state = outputs.last_hidden_state  # (B, token, 768)
        return last_hidden_state
    

    def _forward_dinov2_vitb14(self, x):
        # :arg x: (B, 3, H, W)
        # :return x: (B, token, 768)
        x = self.model.prepare_tokens_with_masks(x)
        for blk in self.model.blocks:
            x = blk(x)
        x = x[:, 1:]  # Remove CLS token, dynamic binding of slots not needed here!
        return x



class VisualEncoder3D(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass


    def forward(self, x):
        pass


class Aggregator(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass


    def forward(self, x):
        pass



if __name__ == "__main__":
    # [ ]: test VisualEncoder2D
    pass 
