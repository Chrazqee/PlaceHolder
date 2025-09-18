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


class Loss_Function(nn.Module):
    def __init__(self, 
                 token_num,
                 num_slots
                 ):
        """ Reconstruction loss function with MSE loss between reconstruction and target.
            e.g. slots after decoding and tokens from pretrained DINO.
        Args:
            token_num: int, number of tokens
            num_slots: int, number of slots

        Returns:
            loss: tensor, reconstruction loss    
        """
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.token_num = token_num
        self.num_slots = num_slots
        self.epsilon = 1e-8

    def forward(self, reconstruction, masks, target):
        # :args reconstruction: (B, token, 768)
        # :args masks:  (B, S, token)
        # :args target: (B, token, 768)
        target = target.detach()
        loss = self.mse(reconstruction, target.detach()).mean()
        return loss
