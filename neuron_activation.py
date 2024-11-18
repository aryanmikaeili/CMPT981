import torch
import torch.nn as nn

class ActivationTrackingMLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, width=256, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, width))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(width, width))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(width, output_dim))
        self.layers = nn.Sequential(*self.layers)
        
        self.relu_indices = [i for i, layer in enumerate(self.layers) if isinstance(layer, nn.ReLU)]
        self.linear_indices = [i for i, layer in enumerate(self.layers) if isinstance(layer, nn.Linear)]

    def forward(self, x, track_activations=False):
        if not track_activations:
            return torch.sigmoid(self.layers(x))
            
        activations = {}
        current = x
        
        for i, layer in enumerate(self.layers):
            current = layer(current)
            if i in self.relu_indices:
                activations[f'relu_{i}'] = current.clone()
        
        output = torch.sigmoid(current)
        return output, activations

    def reinitialize_neurons(self, X, threshold, reinit_input=True, reinit_output=True):
        """
        Reinitialize neurons based on their average activation over samples X.
        
        Args:
            X (torch.Tensor): Input samples to compute activations
            threshold (float): Threshold for average activation
            reinit_input (bool): Whether to reinitialize input weights
            reinit_output (bool): Whether to reinitialize output weights
        """
        # Get activations for all samples
        _, activations = self(X, track_activations=True)
        
        total_neurons = 0
        total_reinitialized = 0
        
        # For each ReLU layer
        for relu_idx, linear_idx in zip(self.relu_indices[:-1], self.linear_indices[1:-1]):
            # Get average activation for each neuron over all samples
            relu_activations = activations[f'relu_{relu_idx}']
            avg_activations = relu_activations.mean(dim=0)  # Average over batch dimension
            
            # Find neurons to reinitialize
            neurons_to_reinit = avg_activations > threshold
            num_reinit = neurons_to_reinit.sum().item()
            
            total_neurons += len(avg_activations)
            total_reinitialized += num_reinit
            
            if num_reinit > 0:
                current_layer = self.layers[linear_idx]
                next_layer = self.layers[linear_idx + 2]  # Skip ReLU to get next linear layer
                
                # Reinitialize input weights if requested
                if reinit_input:
                    nn.init.kaiming_normal_(current_layer.weight[neurons_to_reinit])
                    if current_layer.bias is not None:
                        nn.init.zeros_(current_layer.bias[neurons_to_reinit])
                
                # Reinitialize output weights if requested
                if reinit_output:
                    nn.init.kaiming_normal_(next_layer.weight[:, neurons_to_reinit])
                
                print(f"Layer {linear_idx}: Reinitialized {num_reinit}/{len(avg_activations)} neurons "
                      f"({(num_reinit/len(avg_activations)*100):.1f}%)")
        
        print(f"\nTotal: Reinitialized {total_reinitialized}/{total_neurons} neurons "
              f"({(total_reinitialized/total_neurons*100):.1f}%)")
        
        return total_reinitialized, total_neurons

def test_batch_processing():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create model with smaller width for better visualization
    model = ActivationTrackingMLP(input_dim=2, output_dim=3, width=4, num_layers=2)
    
    # Generate batch of inputs (batch_size=3)
    batch_input = torch.tensor([
        [0.5, -0.2],  # sample 1
        [-0.1, 0.7],  # sample 2
        [0.3, 0.4]    # sample 3
    ], dtype=torch.float32)
    
    print("Input batch:")
    print(batch_input)
    print("\nInput shape:", batch_input.shape)
    
    # Get output and activations
    output, activations = model(batch_input, track_activations=True)
    
    # Print activations for each ReLU layer
    print("\nReLU Activations:")
    for name, activation in activations.items():
        print(f"\n{name}")
        print(f"Shape: {activation.shape}")
        print("Values:")
        print(activation)
    
    print("\nFinal Output:")
    print(f"Shape: {output.shape}")
    print("Values:")
    print(output)

def test_reinitialization():
    # Set random seed
    torch.manual_seed(42)
    
    # Create model and data
    model = ActivationTrackingMLP(input_dim=2, output_dim=3, width=10, num_layers=2)
    X = torch.randn(100, 2)  # 100 samples
    
    print("Testing reinitialization with different settings:\n")
    
    print("1. Reinitialize both input and output weights (threshold=0.5):")
    model.reinitialize_neurons(X, threshold=0.5, reinit_input=True, reinit_output=True)
    
    print("\n2. Reinitialize only input weights (threshold=0.7):")
    model.reinitialize_neurons(X, threshold=0.7, reinit_input=True, reinit_output=False)
    
    print("\n3. Reinitialize only output weights (threshold=0.3):")
    model.reinitialize_neurons(X, threshold=0.3, reinit_input=False, reinit_output=True)

if __name__ == "__main__":
    test_batch_processing()
    test_reinitialization()