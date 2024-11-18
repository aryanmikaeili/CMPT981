import torch
import torch.nn as nn


class PE(nn.Module):
    def __init__(self, num_res = 6):
        super(PE, self).__init__()
        self.num_res = num_res
    def forward(self, x):
        outs = [x]
        for r in range(self.num_res):
            outs.append(torch.sin(x * 2 ** r))
            outs.append(torch.cos(x * 2 ** r))

        out = torch.cat(outs, dim=-1)
        return out
class MLP(nn.Module):
    def __init__(self, input_dim = 2, output_dim = 3, width = 256, num_layers = 2):

        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, width))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(width, width))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(width, output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.layers(x)
        out = torch.sigmoid(out)
        return out


class FCNet(nn.Module):
    def __init__(self, use_pe=True, num_res = 6, num_layers = 3, width = 256):
        super(FCNet, self).__init__()
        input_dim = 2
        if use_pe:
            num_res = num_res
            self.pe = PE(num_res=num_res)
            input_dim = 4 * num_res + 2
        self.use_pe =  use_pe
        self.mlp = MLP(input_dim, 3, width, num_layers)

    def forward(self, x):
        if self.use_pe:
            x = self.pe(x)
        out = self.mlp(x)
        return out

