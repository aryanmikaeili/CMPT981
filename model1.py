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
    def __init__(self, input_dim, output_dim, width, num_layers):
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
        #out = torch.sigmoid(out)
        return out


class FCNet(nn.Module):
    def __init__(self, use_pe=True, num_res = 6, num_layers = 3, width = 256,
                 high_freq_width=128, high_freq_layers=2,
                 low_freq_width=256, low_freq_layers=3):
        super(FCNet, self).__init__()
        input_dim = 2
        if use_pe:
            num_res = num_res
            self.pe = PE(num_res=num_res)
            input_dim = 4 * num_res + 2
        self.use_pe =  use_pe
        self.input_dim = input_dim
        print(f"in dim: {input_dim}, num_layers: {num_layers}, width: {width}")

        # Store per-branch shapes so reset_*_freq rebuilds identical sub-networks.
        self.high_freq_in = input_dim - 2
        self.high_freq_width = high_freq_width
        self.high_freq_layers = high_freq_layers
        self.low_freq_in = 2
        self.low_freq_width = low_freq_width
        self.low_freq_layers = low_freq_layers

        self.high_freq_mlp = MLP(self.high_freq_in, 3, self.high_freq_width, num_layers=self.high_freq_layers)
        self.low_freq_mlp = MLP(self.low_freq_in, 3, self.low_freq_width, num_layers=self.low_freq_layers)


    def forward(self, x):
        if self.use_pe:
            x = self.pe(x)
        high_freq = self.high_freq_mlp(x[:, 2:])
        low_freq = self.low_freq_mlp(x[:, :2])
        out = high_freq + low_freq
        out = torch.sigmoid(out)
        return out, torch.sigmoid(low_freq), torch.sigmoid(high_freq)

    def _device(self):
        # Follow whatever device the model already lives on instead of hardcoding cuda.
        return next(self.parameters()).device

    def reset_high_freq(self):
        self.high_freq_mlp = MLP(self.high_freq_in, 3, self.high_freq_width,
                                 num_layers=self.high_freq_layers).to(self._device())

    def reset_low_freq(self):
        self.low_freq_mlp = MLP(self.low_freq_in, 3, self.low_freq_width,
                                num_layers=self.low_freq_layers).to(self._device())
        

