import torch
import torch.nn as nn

# Your MLP class definition (as you provided)
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

# Create an instance of the MLP
model = MLP()

# Print out layer details
print("Model Architecture:")
print(model)

# Inspect weight initialization
print("\nLayer Weight Initialization:")
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"{name} shape: {param.shape}")
        print(f"Weight statistics:")
        print(f"  Mean: {param.data.mean().item()}")
        print(f"  Std: {param.data.std().item()}")
        print(f"  Min: {param.data.min().item()}")
        print(f"  Max: {param.data.max().item()}")
        print()

# Visualize weight distribution (optional)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
for name, param in model.named_parameters():
    if 'weight' in name:
        plt.hist(param.data.numpy().flatten(), bins=50, alpha=0.5, label=name)

plt.title("Weight Distribution")
plt.xlabel("Weight Values")
plt.ylabel("Frequency")
plt.legend()
plt.show()