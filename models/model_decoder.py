import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, 
                 slot_dim,
                 hidden_dim=2048,
                 ):
        super().__init__()

        # === Token calculations ===
        self.slot_dim =slot_dim
        self.hidden_dim = hidden_dim

        # === MLP Based Decoder ===
        self.layer1 = nn.Linear(self.slot_dim, self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer4 = nn.Linear(self.hidden_dim, 768 + 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, slot_maps):
        # :arg slot_maps: (B * S, token, D_slot)

        slot_maps = self.relu(self.layer1(slot_maps))    # (B * S, token, D_hidden)
        slot_maps = self.relu(self.layer2(slot_maps))    # (B * S, token, D_hidden)
        slot_maps = self.relu(self.layer3(slot_maps))    # (B * S, token, D_hidden)

        slot_maps = self.layer4(slot_maps)               # (B * S, token, 768 + 1)

        return slot_maps
