import torch
from torch.nn import Linear, Sequential, ReLU, Sigmoid, BCELoss
from torch.optim import Adam

class CellEdgeClassifier(torch.nn.Module):
    def __init__(self):
        super(CellEdgeClassifier, self).__init__()
        self.mlp = Sequential(
            Linear(12, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 16),
            ReLU(),
            Linear(16, 1),
            Sigmoid() 
        )

    def forward(self, x, edge_index, edge_attr):
        src_nodes, dst_nodes = edge_index
        x_src = x[src_nodes]
        x_dst = x[dst_nodes]
        edge_inputs = torch.cat([x_src, x_dst, edge_attr], dim=1)
        return self.mlp(edge_inputs)

def train_model(model, data, epochs=100):
    print("--- Generating Training Labels ---")
    # For this initial test run, we will generate pseudo-labels based on normalized distance.
    # We will assume very close edges (normalized distance < -0.5) are True (1.0), and others are False (0.0).
    distances = data.edge_attr[:, 0] 
    labels = (distances < -0.5).float().view(-1, 1)
    
    criterion = BCELoss() # Binary Cross Entropy Loss
    optimizer = Adam(model.parameters(), lr=0.005) # Adam Optimizer
    
    print("\n--- Starting Training Loop ---")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Forward pass: Make predictions
        predictions = model(data.x, data.edge_index, data.edge_attr)
        
        # 2. Calculate Loss: How far off were the predictions?
        loss = criterion(predictions, labels)
        
        # 3. Backward pass: Update the weights to get smarter
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {loss.item():.4f}")
            
    return model, predictions, labels

def main():
    print("--- Loading PyTorch Graph Data ---")
    try:
        data = torch.load('graph_data.pt', weights_only=False)
    except FileNotFoundError:
        print("Error: 'graph_data.pt' not found.")
        return

    model = CellEdgeClassifier()
    
    # Run the training loop!
    trained_model, final_preds, targets = train_model(model, data, epochs=100)
    
    print("\n--- Training Complete ---")
    print("Sample final predictions vs targets:")
    for i in range(5):
        pred_val = final_preds[i].item()
        target_val = targets[i].item()
        print(f"Edge {i}: Predicted = {pred_val:.4f} | Target = {target_val}")

    # Save the trained model weights
    torch.save(trained_model.state_dict(), 'gnn_tracker_weights.pth')
    print("\nSaved trained model to 'gnn_tracker_weights.pth'")

if __name__ == "__main__":
    main()