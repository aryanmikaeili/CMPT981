import torch
import torch.nn as nn
from neuron_activation import ActivationTrackingMLP

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

class FCNet(nn.Module):
    def __init__(self, use_pe=True, num_res=6, num_layers=3, width=256):
        super(FCNet, self).__init__()
        input_dim = 2
        if use_pe:
            num_res = num_res
            self.pe = PE(num_res=num_res)
            input_dim = 4 * num_res + 2
        self.use_pe = use_pe
        self.mlp = ActivationTrackingMLP(input_dim, 3, width, num_layers)

    def forward(self, x, track_activations=False):
        if self.use_pe:
            x = self.pe(x)
        
        if not track_activations:
            out = self.mlp(x)
            return out
        else:
            out, activations = self.mlp(x, track_activations)
            return out, activations
            
    def reinitialize_neurons(self, X, threshold = None, top_percentage = None, reinit_input=True, reinit_output=True, X_negative = None):
        """
        Reinitialize neurons of the MLP based on their average activation over samples X.
        
        Args:
            X (torch.Tensor): Input samples to compute activations
            threshold (float): Threshold for average activation
            top_percentage (float, optional): Percentage of qualifying neurons to reinitialize (0-100)
            reinit_input (bool): Whether to reinitialize input weights
            reinit_output (bool): Whether to reinitialize output weights
        """
        # First apply positional encoding if used
        if self.use_pe:
            X = self.pe(X)
            if X_negative is not None:
                X_negative = self.pe(X_negative)
        # Call the MLP's reinitialization method
        
        return self.mlp.reinitialize_neurons(X= X, threshold= threshold, top_percentage= top_percentage,  
                                             reinit_input= reinit_input, reinit_output= reinit_output, X_negative= X_negative)
