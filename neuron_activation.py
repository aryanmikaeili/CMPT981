import torch
import torch.nn as nn
<<<<<<< HEAD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

=======
import math 
>>>>>>> e05ca4696ded760dc73db1dc584c4df407bb92d7
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
            current = layer(current.to(device))
            if i in self.relu_indices:
                activations[f'relu_{i}'] = current.clone()
        
        output = torch.sigmoid(current)
        return output, activations

    def reinitialize_neurons(self, X, threshold=None, top_percentage=None, reinit_input=True, reinit_output=True, X_negative=None):
        """
        Reinitialize neurons based on their activation patterns. If X_negative is provided,
        uses contrast scores (ratio of positive to negative activation) instead of raw activations.
        
        Args:
            X (torch.Tensor): Desired input samples to compute activations
            threshold (float, optional): Threshold for contrast score or activation
            top_percentage (float, optional): Percentage of neurons to reinitialize (0-100)
            reinit_input (bool): Whether to reinitialize input weights
            reinit_output (bool): Whether to reinitialize output weights
            X_negative (torch.Tensor, optional): If provided, select neurons based on contrast score
        """
        if threshold is None and top_percentage is None:
            raise ValueError("At least one of 'threshold' or 'top_percentage' must be provided")
            
<<<<<<< HEAD
        # Get activations for all samples
        _, activations = self(X, track_activations=True)
        # print(activations)
        # for key, value in activations.items():
        #     print(f'{key}: {value}')
        #     print(f'mean of {key}: {value.mean()}')

=======
        # Get activations for positive samples
        _, activations_pos = self(X, track_activations=True)
        
        # Get activations for negative samples if provided
        if X_negative is not None:
            _, activations_neg = self(X_negative, track_activations=True)
>>>>>>> e05ca4696ded760dc73db1dc584c4df407bb92d7
        
        total_neurons = 0
        total_reinitialized = 0
        
        # For each ReLU layer
        for relu_idx, linear_idx in zip(self.relu_indices[:-1], self.linear_indices[1:-1]):
            # Get average activation for positive samples
            relu_activations_pos = activations_pos[f'relu_{relu_idx}']
            avg_activations_pos = relu_activations_pos.mean(dim=0)
            
            if X_negative is not None:
                # Get average activation for negative samples
                relu_activations_neg = activations_neg[f'relu_{relu_idx}']
                avg_activations_neg = relu_activations_neg.mean(dim=0)
                
                # Calculate contrast score: ratio of positive to negative activation
                eps = 1e-6  # small epsilon to avoid division by zero
                scores = avg_activations_pos / (avg_activations_neg + eps)
                
                # Print statistics about contrast scores
                print(f"\nLayer {linear_idx} contrast scores:")
                print(f"Min: {scores.min():.2f}, Max: {scores.max():.2f}, "
                      f"Mean: {scores.mean():.2f}, Median: {scores.median():.2f}")
            else:
                scores = avg_activations_pos
           
            if threshold is None:
                # Only top percentage-based
                k = int(len(scores) * top_percentage)
                if k > 0:
                    _, top_indices = torch.topk(scores, k)
                    neurons_to_reinit = torch.zeros_like(scores, dtype=torch.bool)
                    neurons_to_reinit[top_indices] = True
                else:
                    neurons_to_reinit = torch.zeros_like(scores, dtype=torch.bool)
            
            elif top_percentage is None:
                # Only threshold-based
                neurons_to_reinit = scores > threshold
                
            else:
                # Combined case: top percentage of neurons above threshold
                above_threshold = scores > threshold
                if above_threshold.sum() > 0:
                    # Get values and indices of neurons above threshold
                    qualified_values = scores[above_threshold]
                    qualified_indices = torch.where(above_threshold)[0]
                    
                    # Calculate how many to reinitialize
                    k = int(len(qualified_indices) * top_percentage)
                    if k > 0:
                        # Get indices of top k among qualified neurons
                        _, top_k_indices = torch.topk(qualified_values, k)
                        selected_indices = qualified_indices[top_k_indices]
                        
                        # Create final mask
                        neurons_to_reinit = torch.zeros_like(scores, dtype=torch.bool)
                        neurons_to_reinit[selected_indices] = True
                    else:
                        neurons_to_reinit = torch.zeros_like(scores, dtype=torch.bool)
                else:
                    neurons_to_reinit = torch.zeros_like(scores, dtype=torch.bool)
            
            num_reinit = neurons_to_reinit.sum().item()
            total_neurons += len(scores)
            total_reinitialized += num_reinit
            
            if num_reinit > 0:
                current_layer = self.layers[linear_idx]
                next_layer = self.layers[linear_idx + 2]  # Skip ReLU to get next linear layer
                
                # Reinitialize input weights if requested
                if reinit_input:
                    with torch.no_grad():
                        device = current_layer.weight.device
                        current_layer.weight.data[neurons_to_reinit] = nn.init.kaiming_uniform_(
                            current_layer.weight.data[neurons_to_reinit].clone(), a=math.sqrt(5)).to(device)
                        if current_layer.bias is not None:
                            nn.init.zeros_(current_layer.bias[neurons_to_reinit]).to(device)
                
                # Reinitialize output weights if requested
                if reinit_output:
                    with torch.no_grad():
                        device = next_layer.weight.device
                        next_layer.weight[:, neurons_to_reinit] = nn.init.kaiming_uniform_(
                            next_layer.weight[:, neurons_to_reinit].clone(), a=math.sqrt(5)).to(device)
                
                # Print statistics
                if X_negative is not None:
                    print(f"Selected neurons were based on contrast scores")
                    
                
                if threshold is not None and top_percentage is not None:
                    print(f"Layer {linear_idx}: {(above_threshold.sum().item()/len(scores)*100):.1f}% neurons above threshold {threshold:.3f}")
                    print(f"           Reinitialized top {top_percentage}% of them: {num_reinit}/{len(scores)} neurons "
                            f"({(num_reinit/len(scores)*100):.1f}%)")
                elif threshold is not None:
                    print(f"Layer {linear_idx}: Reinitialized {num_reinit}/{len(scores)} neurons "
                            f"({(num_reinit/len(scores)*100):.1f}%) with activation > {threshold:.3f}")
                else:
                    print(f"Layer {linear_idx}: Reinitialized top {num_reinit}/{len(scores)} neurons "
                            f"({(num_reinit/len(scores)*100):.1f}%)")
    
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