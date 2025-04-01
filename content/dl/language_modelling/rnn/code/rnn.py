import torch
import torch.nn as nn
from torch import Tensor

class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(input_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(hidden_size))
        self.weight_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(hidden_size))

    def forward(self, input:Tensor, hx:Tensor|None = None):
        B, T, _ = input.shape
        if hx is None:
            hx = torch.zeros(B, self.hidden_size, device=input.device, dtype=input.dtype)

        outputs = []
        for i in range(T):
            out_preact = input[:,i,:] @ self.weight_ih + self.bias_ih + hx @ self.weight_hh + self.bias_hh
            hx = torch.tanh(out_preact)
            outputs.append(hx)

        return torch.stack(outputs, dim=1), hx
        
