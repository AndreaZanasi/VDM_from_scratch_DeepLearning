import torch
from torch import nn

class NoiseSchedule(nn.Module):
    def __init__(self, gamma_min=-13.3, gamma_max=5.0):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, t):
        raise NotImplementedError


class LinearSchedule(NoiseSchedule):
    def forward(self, t):
        return self.gamma_min + t * (self.gamma_max - self.gamma_min)


class LearnedSchedule(NoiseSchedule):
    def __init__(self, gamma_min=-13.3, gamma_max=5.0):
        super().__init__(gamma_min, gamma_max)
        
        self.layer1 = nn.Linear(1, 1, bias=True)
        self.layer2 = nn.Linear(1, 1024, bias=False)
        self.layer3 = nn.Linear(1024, 1, bias=False)
        
        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            self.layer1.weight.abs_()
            self.layer2.weight.abs_()
            self.layer3.weight.abs_()

    def forward(self, t):
        t = t.view(-1, 1)
        
        gamma_tilde = self._forward_tilde(t)
        gamma_tilde_0 = self._forward_tilde(torch.zeros_like(t))
        gamma_tilde_1 = self._forward_tilde(torch.ones_like(t))
        
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * (
            (gamma_tilde - gamma_tilde_0) / (gamma_tilde_1 - gamma_tilde_0)
        )
        
        return gamma.squeeze(-1)

    def _forward_tilde(self, t):
        w1 = self.layer1.weight.clamp(min=0)
        w2 = self.layer2.weight.clamp(min=0)
        w3 = self.layer3.weight.clamp(min=0)
        b1 = self.layer1.bias
        
        h1 = nn.functional.linear(t, w1, b1)
        h2 = nn.functional.linear(h1, w2)
        h2 = torch.sigmoid(h2)
        h3 = nn.functional.linear(h2, w3)
        
        return h1 + h3