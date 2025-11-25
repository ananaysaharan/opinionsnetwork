import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from surrogate_model import SurrogateModel

def train_model(epochs=100, learning_rate=0.001):
    print("--- Day 18: The Training Loop ---")
    
    # 1. Load Data
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    X_path = os.path.join(output_dir, 'surrogate_X.pt')
    Y_path = os.path.join(output_dir, 'surrogate_Y.pt')
    
    if not os.path.exists(X_path) or not os.path.exists(Y_path):
        print("Error: Training data not found. Please run Day 16 script first.")
        return

    print("Loading data...")
    X = torch.load(X_path)
    Y = torch.load(Y_path)
    
    # Create DataLoader for batching (optional but good practice, though user didn't explicitly ask for batches, 
    # doing full batch or mini-batch is fine. Let's do mini-batches for better convergence)
    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. Initialize Model, Loss, Optimizer
    model = SurrogateModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Model initialized. Training for {epochs} epochs...")
    
    start_time = time.time()
    loss_history = []
    
    # 3. Training Loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_Y in dataloader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            
            # Compute loss
            loss = criterion(outputs, batch_Y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)
        
        # Average loss for the epoch
        epoch_loss /= len(dataset)
        loss_history.append(epoch_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")
            
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.2f} seconds.")
    print(f"Final Loss: {loss_history[-1]:.6f}")
    
    # Save the trained model
    model_save_path = os.path.join(output_dir, 'surrogate_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_model()
