import sys
import os
import torch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from surrogate_model import SurrogateModel

def verify_architecture():
    print("--- Day 17: The Architecture (LSTM) ---")
    
    model = SurrogateModel()
    print("\nModel Architecture:")
    print(model)
    
    # Verify with dummy input
    batch_size = 5
    dummy_input = torch.randn(batch_size, 2)
    print(f"\nDummy Input Shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output Shape: {output.shape}")
    
    expected_shape = (batch_size, 100)
    if output.shape == expected_shape:
        print("\nSUCCESS: Output shape matches requirements (Batch, 100).")
    else:
        print(f"\nFAILURE: Expected {expected_shape}, got {output.shape}")

if __name__ == "__main__":
    verify_architecture()
