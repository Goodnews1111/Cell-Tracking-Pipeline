import torch
from torch.nn import Linear, Sequential, ReLU, Sigmoid
import torch.nn.functional as F

class CellEdgeClassifier(torch.nn.Module):
    def __init__(self):
        super(CellEdgeClassifier, self).__init__()
        # Input features: Source Node (4) + Target Node (4) + Edge Attr (4) = 12 features
        self.mlp = Sequential(
            Linear(12, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 16),
            ReLU(),
            Linear(16, 1),
            Sigmoid() # Outputs a probability between 0.0 and 1.0
        )

    def forward(self, x, edge_index, edge_attr):
        # 1. Get the indices for the source and target nodes of every edge
        src_nodes, dst_nodes = edge_index
        
        # 2. Extract the actual node features using those indices
        x_src = x[src_nodes]
        x_dst = x[dst_nodes]
        
        # 3. Concatenate everything together into a massive feature matrix for the edges
        edge_inputs = torch.cat([x_src, x_dst, edge_attr], dim=1)
        
        # 4. Pass through the Neural Network
        return self.mlp(edge_inputs)

def main():
    print("--- Loading PyTorch Graph Data ---")
    try:
        data = torch.load('graph_data.pt')
    except FileNotFoundError:
        print("Error: 'graph_data.pt' not found. Run build_datasets.py first.")
        return

    # Initialize the Neural Network
    model = CellEdgeClassifier()
    print("\n--- Neural Network Architecture ---")
    print(model)
    
    # Run a test Forward Pass (untrained)
    print("\n--- Testing Forward Pass ---")
    with torch.no_grad(): # Disable gradients for a quick test
        predictions = model(data.x, data.edge_index, data.edge_attr)
        
    print(f"Generated {len(predictions)} edge probabilities.")
    print(f"Sample prediction outputs (Probabilities):\n{predictions[:5].flatten().numpy()}")

if __name__ == "__main__":
    main()