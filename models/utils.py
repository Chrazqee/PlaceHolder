import math

import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(
            self, 
            input_dim, 
            hidden_dim, 
            output_dim, 
            residual=False, 
            dropout=0.1,
            layer_order="None"
            ):
        """ A simple 2-layer MLP with optional residual connection and layer normalization.
        Args:
            input_dim: int, input feature dimension
            hidden_dim: int, hidden layer dimension
            output_dim: int, output feature dimension
            residual: bool, whether to use residual connection (default: False)
            layer_order: str, order of layer normalization, can be "pre", "post", or "none" (default: "none")
                 - "pre": layer normalization before each linear layer
                 - "post": layer normalization after each linear layer

        Assertions:
            - if residual is True, input_dim must be equal to output_dim
            - layer_order in ["pre", "post"], if layer_order is "none", will trigger assertion error
                 
        returns:
            output: tensor, output feature with dimension output_dim
        """
        super().__init__()
        self.residual = residual
        self.layer_order = layer_order
        if residual:
            assert input_dim == output_dim

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

        if layer_order in ["pre", "post"]:
            self.norm = nn.LayerNorm(input_dim)
        else:
            assert layer_order == "none"

    def forward(self, x):
        input = x

        if self.layer_order == "pre":
            x = self.norm(x)

        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.dropout(x)

        if self.residual:
            x = x + input
        if self.layer_order == "post":
            x = self.norm(x)

        return x
    

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super(PositionEmbeddingSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)



class PositionEmbeddingSine3D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, add_temporal=True):
        super(PositionEmbeddingSine3D, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.add_temporal = add_temporal
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        # b, t, c, h, w
        assert x.dim() == 5, f"{x.shape} should be a 5-dimensional Tensor, got {x.dim()}-dimensional Tensor instead"
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(1), x.size(3), x.size(4)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        z_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        dim_t_z = torch.arange((self.num_pos_feats * 2), dtype=torch.float32, device=x.device)
        dim_t_z = self.temperature ** (2 * (dim_t_z // 2) / (self.num_pos_feats * 2))

        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t_z
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        if not self.add_temporal:
            pos_z = 0.0
        pos = (torch.cat((pos_y, pos_x), dim=4) + pos_z).permute(0, 1, 4, 2, 3)  # b, t, c, h, w
        return pos


# [ ]: check
class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        self.orig_ch = channels
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor, input_range=None):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        pos_x, pos_y, pos_z = tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]
        sin_inp_x = torch.einsum("bi,j->bij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("bi,j->bij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("bi,j->bij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)

        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)

        emb = torch.cat((emb_x, emb_y, emb_z), dim=-1)
        return emb[:, :, :self.orig_ch].permute((0, 2, 1))



