import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
            self, 
            input_dim, 
            hidden_dim, 
            output_dim, 
            residual=False, 
            layer_order="none"
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
        self.dropout = nn.Dropout(p=0.1)

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
    

class Loss_Function(nn.Module):
    def __init__(self, 
                 token_num,
                 num_slots
                 ):
        """ Reconstruction loss function with MSE loss.
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
