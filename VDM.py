import torch
import numpy as np
from torch import nn, sigmoid, sqrt, exp
from torch.special import expm1

def get_alpha(gamma):
    return sqrt(sigmoid(-gamma))

def get_sigma(gamma):
    return sqrt(sigmoid(gamma))

def get_snr(gamma):
    return exp(-gamma)

class LearnedSchedule(nn.Module):
    def __init__(self, gamma_min=-13.3, gamma_max=5.0):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        
        self.l1 = nn.Linear(1, 1, bias=True)
        self.l2 = nn.Linear(1, 1024, bias=False)
        self.l3 = nn.Linear(1024, 1, bias=False)
        
        with torch.no_grad():
            self.l1.weight.abs_()
            self.l2.weight.abs_()
            self.l3.weight.abs_()

    def forward(self, t):
        t = t.view(-1, 1)
        
        gamma_tilde = self.forward_tilde(t)
        
        gamma_tilde_0 = self.forward_tilde(torch.zeros_like(t))
        gamma_tilde_1 = self.forward_tilde(torch.ones_like(t))
        
        gamma_t = self.gamma_min + (self.gamma_max - self.gamma_min) * (
            (gamma_tilde - gamma_tilde_0) / (gamma_tilde_1 - gamma_tilde_0)
        )
        
        return gamma_t.squeeze(-1)
    
    def forward_tilde(self, t):
        # Use functional linear with clamped weights to enforce monotonicity
        # without modifying weights in-place during forward pass
        w1 = self.l1.weight.clamp(min=0)
        b1 = self.l1.bias
        w2 = self.l2.weight.clamp(min=0)
        w3 = self.l3.weight.clamp(min=0)
        
        l1_out = torch.nn.functional.linear(t, w1, b1)
        l2_out = torch.nn.functional.linear(l1_out, w2)
        l2_out = torch.sigmoid(l2_out)
        l3_out = torch.nn.functional.linear(l2_out, w3)
        return l1_out + l3_out

class ForwardDiffusion(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def sample_z(self, x, t, noise=None):
        gamma_t = self.gamma(t)
        gamma_t_padded = gamma_t.view(-1, *([1] * (x.ndim - 1)))
        
        alpha_t = get_alpha(gamma_t_padded)
        sigma_t = get_sigma(gamma_t_padded)
        
        if noise is None:
            noise = torch.randn_like(x)
        
        z_t = alpha_t * x + sigma_t * noise
        return z_t, gamma_t, noise

class ReverseDiffusion(nn.Module):
    def __init__(self, model, gamma):
        super().__init__()
        self.model = model
        self.gamma = gamma
    
    @torch.no_grad()
    def sample(self, batch_size, image_shape, T=1000, device='cuda'):
        z = torch.randn((batch_size, *image_shape), device=device)
        steps = torch.linspace(1.0, 0.0, T + 1, device=device)
        
        for i in range(T):
            t = steps[i].expand(batch_size)
            s = steps[i + 1].expand(batch_size)
            
            gamma_t = self.gamma(t)
            gamma_s = self.gamma(s)
            
            gamma_t_padded = gamma_t.view(-1, *([1] * (len(image_shape) + 1)))
            gamma_s_padded = gamma_s.view(-1, *([1] * (len(image_shape) + 1)))
            
            alpha_t = get_alpha(gamma_t_padded)
            alpha_s = get_alpha(gamma_s_padded)
            sigma_t = get_sigma(gamma_t_padded)
            sigma_s = get_sigma(gamma_s_padded)
            
            eps_hat = self.model(z, gamma_t)
            
            c = -expm1(gamma_s_padded - gamma_t_padded)
            mean = (alpha_s / alpha_t) * (z - sigma_t * c * eps_hat)
            
            if i < T - 1:
                noise = torch.randn_like(z)
                z = mean + sigma_s * sqrt(c) * noise
            else:
                z = mean
        
        z = torch.clamp((z + 1) / 2, 0, 1)
        return z

class VDM(nn.Module):    
    def __init__(self, model, gamma_min=-13.3, gamma_max=5.0, 
                 vocab_size=256, T=1000, device='cuda'):
        super().__init__()
        self.model = model
        self.vocab_size = vocab_size
        self.device = device
        self.T = T
        
        self.gamma = LearnedSchedule(gamma_min, gamma_max)
        self.forward_diffusion = ForwardDiffusion(self.gamma)
        self.reverse_diffusion = ReverseDiffusion(model, self.gamma)
    
    def forward(self, batch):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, _ = batch
        else:
            x = batch
        
        x = x.to(self.device)
        B = x.shape[0]
        
        x_int = torch.round(x * (self.vocab_size - 1)).long()
        x_cont = 2 * ((x_int + 0.5) / self.vocab_size) - 1
        
        i = torch.randint(1, self.T + 1, (B,), device=self.device)
        t = i.float() / self.T
        s = (i - 1).float() / self.T
        
        z_t, gamma_t, eps = self.forward_diffusion.sample_z(x_cont, t)
        eps_hat = self.model(z_t, gamma_t)
        
        gamma_s = self.gamma(s)
        snr_s = get_snr(gamma_s)
        snr_t = get_snr(gamma_t)
        weight = snr_s - snr_t
        
        mse = ((eps - eps_hat) ** 2).mean(dim=(1, 2, 3))
        diffusion_loss = 0.5 * self.T * weight * mse
        
        gamma_1 = self.gamma(torch.ones(B, device=self.device))
        snr_1 = get_snr(gamma_1)
        mean_sq = (snr_1 / (snr_1 + 1)).view(B, 1, 1, 1) * (x_cont ** 2)
        prior_loss = 0.5 * mean_sq.view(B, -1).sum(dim=1)
        
        gamma_0 = self.gamma(torch.zeros(B, device=self.device))
        gamma_0_padded = gamma_0.view(B, 1, 1, 1)
        alpha_0 = get_alpha(gamma_0_padded)
        sigma_0 = get_sigma(gamma_0_padded)
        
        z_0 = (z_t - get_sigma(gamma_t.view(B, 1, 1, 1)) * eps_hat) / get_alpha(gamma_t.view(B, 1, 1, 1))
        
        x_vals = 2 * ((torch.arange(self.vocab_size, device=self.device).float() + 0.5) / self.vocab_size) - 1
        mu_vals = alpha_0.unsqueeze(-1) * x_vals.view(1, 1, 1, 1, -1)
        
        z_0_exp = z_0.unsqueeze(-1)
        dist_sq = ((z_0_exp - mu_vals) ** 2) / (sigma_0.unsqueeze(-1) ** 2)
        logits = -0.5 * dist_sq
        
        log_probs = torch.log_softmax(logits, dim=-1)
        x_int_exp = x_int.unsqueeze(-1)
        reconstruction_loss = -log_probs.gather(-1, x_int_exp).squeeze(-1)
        reconstruction_loss = reconstruction_loss.view(B, -1).sum(dim=1)
        
        loss = diffusion_loss + prior_loss + reconstruction_loss
        
        return loss.mean(), {
            "loss": loss.mean().item(),
            "diffusion": diffusion_loss.mean().item(),
            "prior": prior_loss.mean().item(),
            "reconstruction": reconstruction_loss.mean().item()
        }