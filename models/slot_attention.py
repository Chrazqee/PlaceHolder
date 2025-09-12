import torch
import torch.nn as nn
from torch.nn import init
from models.utils import MLP


class ISA(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()

        self.num_slots = args.num_slots
        self.scale = args.slot_dim ** -0.5
        self.iters = args.slot_att_iter
        self.slot_dim = args.slot_dim
        self.query_opt = args.query_opt

        self.res_h = args.resize_to[0] // args.patch_size
        self.res_w = args.resize_to[1] // args.patch_size
        self.token = int(self.res_h * self.res_w)

        # === abs_grid ===
        self.sigma = 5
        xs = torch.linspace(-1, 1, steps=self.res_w)                                                # (C_x)
        ys = torch.linspace(-1, 1, steps=self.res_h)                                                # (C_y)

        xs, ys = torch.meshgrid(xs, ys, indexing='xy')                                              # (C_x, C_y), (C_x, C_y)
        xs = xs.reshape(1, 1, -1, 1)                                                                # (1, 1, C_x * C_y, 1)
        ys = ys.reshape(1, 1, -1, 1)                                                                # (1, 1, C_x * C_y, 1)
        self.abs_grid = nn.Parameter(torch.cat([xs, ys], dim=-1), requires_grad=False)              # (1, 1, token, 2)
        assert self.abs_grid.shape[2] == self.token

        self.h = nn.Linear(2, self.slot_dim)
        # === === ===

        # === Slot related ===
        if self.query_opt:
            self.slots = nn.Parameter(torch.Tensor(1, self.num_slots, self.slot_dim))
            init.xavier_uniform_(self.slots)
        else:
            self.slots_mu = nn.Parameter(torch.randn(1, 1, self.slot_dim))
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, self.slot_dim))
            init.xavier_uniform_(self.slots_mu)
            init.xavier_uniform_(self.slots_logsigma)

        self.S_s = nn.Parameter(torch.Tensor(1, self.num_slots, 1, 2))  # (1, S, 1, 2)
        self.S_p = nn.Parameter(torch.Tensor(1, self.num_slots, 1, 2))  # (1, S, 1, 2)

        init.normal_(self.S_s, mean=0., std=.02)
        init.normal_(self.S_p, mean=0., std=.02)
        # === === ===

        # === Slot Attention related ===
        self.Q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.norm = nn.LayerNorm(self.slot_dim)
        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.mlp = MLP(self.slot_dim, 4*self.slot_dim, self.slot_dim,
                       residual=True, layer_order="pre")
        # === === ===

        # === Query & Key & Value ===
        self.K = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.V = nn.Linear(self.slot_dim, self.slot_dim, bias=False)

        self.g = nn.Linear(2, self.slot_dim)
        self.f = nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.slot_dim, self.slot_dim))
        # === === ===

        # Note: starts and ends with LayerNorm
        self.initial_mlp = nn.Sequential(nn.LayerNorm(input_dim),
                                         nn.Linear(input_dim, input_dim),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(input_dim, self.slot_dim),
                                         nn.LayerNorm(self.slot_dim))

        self.final_layer = nn.Linear(self.slot_dim, self.slot_dim)


    def get_rel_grid(self, attn):
        # :arg attn: (B, S, token)
        #
        # :return: (B, S, N, D_slot)

        B, S = attn.shape[:2]
        attn = attn.unsqueeze(dim=2)                                            # (B, S, 1, token)

        abs_grid = self.abs_grid.expand(B, S, self.token, 2)                    # (B, S, token, 2)
        
        S_p = torch.einsum('bsjd,bsij->bsd', abs_grid, attn)                    # (B, S, token, 2) x (B, S, 1, token) -> (B, S, 2)
        S_p = S_p.unsqueeze(dim=2)                                              # (B, S, 1, 2)

        values_ss = torch.pow(abs_grid - S_p, 2)                                # (B, S, token, 2)
        S_s = torch.einsum('bsjd,bsij->bsd', values_ss, attn)                   # (B, S, token, 2) x (B, S, 1, token) -> (B, S, 2)
        S_s = torch.sqrt(S_s)                                                   # (B, S, 2)
        S_s = S_s.unsqueeze(dim=2)                                              # (B, S, 1, 2)

        rel_grid = (abs_grid - S_p) / (S_s * self.sigma)                        # (B, S, token, 2)
        rel_grid = self.h(rel_grid)                                             # (B, S, token, D_slot)

        return rel_grid


    def forward(self, inputs):
        # :arg inputs:              (B, token, D)
        #
        # :return slots:            (B, S, D_slot)
        # :return attn:             (B, S, token)

        B, N, D = inputs.shape
        S = self.num_slots
        D_slot = self.slot_dim
        epsilon = 1e-8

        if self.query_opt:
            slots = self.slots.expand(B, S, D_slot)                     # (B, S, D_slot)
        else:
            mu = self.slots_mu.expand(B, S, D_slot)
            sigma = self.slots_logsigma.exp().expand(B, S, D_slot)
            slots = mu + sigma * torch.randn(mu.shape, device=sigma.device, dtype=sigma.dtype)

        slots_init = slots
        inputs = self.initial_mlp(inputs).unsqueeze(dim=1)          # (B, 1, token, D_slot)
        inputs = inputs.expand(B, S, N, D_slot)                     # (B * F, S, N', D_slot)

        abs_grid = self.abs_grid.expand(B, S, self.token, 2)        # (B, S, token, 2)

        assert torch.sum(torch.isnan(abs_grid)) == 0

        S_s = self.S_s.expand(B, S, 1, 2)                           # (B, S, 1, 2)
        S_p = self.S_p.expand(B, S, 1, 2)                           # (B, S, 1, 2)

        for t in range(self.iters + 1):
            # last iteration for S_s and S_p: t = self.iters
            # last meaningful iteration: t = self.iters - 1

            assert torch.sum(torch.isnan(slots)) == 0, f"Iteration {t}"
            assert torch.sum(torch.isnan(S_s)) == 0, f"Iteration {t}"
            assert torch.sum(torch.isnan(S_p)) == 0, f"Iteration {t}"
            
            if self.query_opt and (t == self.iters - 1):
                slots = slots.detach() + slots_init - slots_init.detach()

            slots_prev = slots
            slots = self.norm(slots)

            # === key and value calculation using rel_grid ===
            rel_grid = (abs_grid - S_p) / (S_s * self.sigma)        # (B, S, token, 2)
            k = self.f(self.K(inputs) + self.g(rel_grid))           # (B, S, token, D_slot)
            v = self.f(self.V(inputs) + self.g(rel_grid))           # (B, S, token, D_slot)

            # === Calculate attention ===
            q = self.Q(slots).unsqueeze(dim=-1)                     # (B, S, D_slot, 1)

            dots = torch.einsum('bsdi,bsjd->bsj', q, k)             # (B, S, D_slot, 1) x (B, S, token, D_slot) -> (B, S, token)
            dots *=  self.scale                                     # (B, S, token)
            attn = dots.softmax(dim=1) + epsilon                    # (B, S, token)

            # === Weighted mean ===
            attn = attn / attn.sum(dim=-1, keepdim=True)            # (B, S, token)
            attn = attn.unsqueeze(dim=2)                            # (B, S, 1, token)
            updates = torch.einsum('bsjd,bsij->bsd', v, attn)       # (B, S, token, D_slot) x (B, S, 1, token) -> (B, S, D_slot)

            # === Update S_p and S_s ===
            S_p = torch.einsum('bsjd,bsij->bsd', abs_grid, attn)    # (B, S, token, 2) x (B, S, 1, token) -> (B, S, 2)
            S_p = S_p.unsqueeze(dim=2)                              # (B, S, 1, 2)

            values_ss = torch.pow(abs_grid - S_p, 2)                # (B, S, token, 2)
            S_s = torch.einsum('bsjd,bsij->bsd', values_ss, attn)   # (B, S, token, 2) x (B, S, 1, token) -> (B, S, 2)
            S_s = torch.sqrt(S_s)                                   # (B, S, 2)
            S_s = S_s.unsqueeze(dim=2)                              # (B, S, 1, 2)

            # === Update ===
            if t != self.iters:
                slots = self.gru(
                    updates.reshape(-1, self.slot_dim),
                    slots_prev.reshape(-1, self.slot_dim))

                slots = slots.reshape(B, -1, self.slot_dim)
                slots = self.mlp(slots)

        slots = self.final_layer(slots_prev)                        # (B, S, D_slot)
        attn = attn.squeeze(dim=2)                                  # (B, S, token)

        return slots, attn
    

class SA(nn.Module):
    def __init__(self, args, input_dim):
        
        super().__init__()
        self.num_slots = args.num_slots
        self.scale = args.slot_dim ** -0.5
        self.iters = args.slot_att_iter
        self.slot_dim = args.slot_dim
        self.query_opt = args.query_opt

        # === Slot related ===
        if self.query_opt:
            self.slots = nn.Parameter(torch.Tensor(1, self.num_slots, self.slot_dim))
            init.xavier_uniform_(self.slots)
        else:
            self.slots_mu = nn.Parameter(torch.randn(1, 1, self.slot_dim))
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, self.slot_dim))
            init.xavier_uniform_(self.slots_mu)
            init.xavier_uniform_(self.slots_logsigma)

        # === Slot Attention related ===
        self.Q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.norm = nn.LayerNorm(self.slot_dim)
        self.update_norm = nn.LayerNorm(self.slot_dim)
        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.mlp = MLP(self.slot_dim, 4 * self.slot_dim, self.slot_dim,
                       residual=True, layer_order="pre")
        # === === ===

        # === Query & Key & Value ===
        self.K = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.V = nn.Linear(self.slot_dim, self.slot_dim, bias=False)

        self.f = nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.slot_dim, self.slot_dim))
        # === === ===

        # Note: starts and ends with LayerNorm
        self.initial_mlp = nn.Sequential(nn.LayerNorm(input_dim),
                                         nn.Linear(input_dim, input_dim),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(input_dim, self.slot_dim),
                                         nn.LayerNorm(self.slot_dim))

        self.final_layer = nn.Linear(self.slot_dim, self.slot_dim)

    def forward(self, inputs):
        # :arg inputs:              (B, token, D)
        #
        # :return slots:            (B, S, D_slot)

        B = inputs.shape[0]
        S = self.num_slots
        D_slot = self.slot_dim
        epsilon = 1e-8

        if self.query_opt:
            slots = self.slots.expand(B, S, D_slot)          # (B, S, D_slot)
        else:
            mu = self.slots_mu.expand(B, S, D_slot)
            sigma = self.slots_logsigma.exp().expand(B, S, D_slot)
            slots = mu + sigma * torch.randn(mu.shape, device=sigma.device, dtype=sigma.dtype)

        slots_init = slots
        inputs = self.initial_mlp(inputs)                    # (B, token, D_slot)

        keys = self.K(inputs)                                # (B, token, D_slot)
        values = self.V(inputs)                              # (B, token, D_slot)
        
        for t in range(self.iters):
            assert torch.sum(torch.isnan(slots)) == 0, f"Iteration {t}"
            
            if t == self.iters - 1 and self.query_opt:
                slots = slots.detach() + slots_init - slots_init.detach()

            slots_prev = slots
            slots = self.norm(slots)
            queries = self.Q(slots)                                     # (B, S, D_slot)

            dots = torch.einsum('bsd,btd->bst', queries, keys)          # (B, S, token)
            dots *= self.scale                                          # (B, S, token)
            attn = dots.softmax(dim=1) + epsilon                        # (B, S, token)
            attn = attn / attn.sum(dim=-1, keepdim=True)                # (B, S, token)

            updates = torch.einsum('bst,btd->bsd', attn, values)        # (B, S, D_slot)

            slots = self.gru(
                    updates.reshape(-1, self.slot_dim),
                    slots_prev.reshape(-1, self.slot_dim))

            slots = slots.reshape(B, -1, self.slot_dim)
            slots = self.mlp(slots)

        self.final_layer(slots)

        return slots

