import torch
import torch.nn as nn

class SurrogateModel(nn.Module):
    def __init__(self):
        super(SurrogateModel, self).__init__()
        # Layer 1: Expand parameters
        self.fc_in = nn.Linear(2, 64)
        
        # Layer 2: The memory (LSTM)
        # input_size=64 matches fc_in output
        # hidden_size=256 (Increased for optimization)
        self.lstm = nn.LSTM(input_size=64, hidden_size=256, batch_first=True)
        
        # Layer 3: Project to time series
        # Projects the 256-dim hidden state to 100 time steps
        self.fc_out = nn.Linear(256, 100)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2). 
                              Contains (c, mu).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 100).
                          Predicted opinion trajectory.
        """
        # 1. Expand parameters
        # x: (batch, 2) -> (batch, 64)
        x = self.fc_in(x)
        
        # 2. Reshape for LSTM
        # LSTM expects (batch, seq_len, input_size) if batch_first=True
        # We treat this as a sequence of length 1
        x = x.unsqueeze(1) # (batch, 1, 64)
        
        # 3. LSTM Pass
        # out: (batch, seq_len, hidden_size) -> (batch, 1, 128)
        # _ : (h_n, c_n)
        out, _ = self.lstm(x)
        
        # 4. Take the output of the last (and only) time step
        out = out[:, -1, :] # (batch, 128)
        
        # 5. Project to 100 time steps
        out = self.fc_out(out) # (batch, 100)
        
        return out
