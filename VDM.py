import torch
import numpy as np
from torch import allclose, argmax, autograd, exp, linspace, nn, sigmoid, sqrt
from torch.special import expm1

def get_alpha(gamma):
    return sqrt(sigmoid(-gamma))

def get_sigma(gamma):
    return sqrt(sigmoid(gamma))

def get_snr(gamma):
    return exp(-gamma)

class ForwardDiffusion(nn.Module):
    def __init__(self, gamma_min, gamma_max, batch_size, device):
        super().__init__()
        self.gamma = LearnedSchedule(gamma_min, gamma_max)
        self.batch_size = batch_size
        self.device = device

    def sample_z(self, x, t, noise=None):
        with torch.enable_grad():
            gamma_t = self.gamma(t)

        gamma_t_padded = gamma_t.view(-1, *([1] * (x.ndim - 1)))
        sigma_t = get_sigma(gamma_t_padded)
        alpha_t = get_alpha(gamma_t_padded)
        mean = alpha_t * x
        if noise is None:
            noise = torch.randn_like(x)

        return mean + noise * sigma_t, gamma_t, noise

class ReverseDiffusion(nn.Module):
    def __init__(self, model, gamma, device):
        super().__init__()
        self.model = model
        self.gamma = gamma
        self.device = device
    
    @torch.no_grad()
    def ancestral_sampling(self, z, t, s):
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        c = -expm1(gamma_s - gamma_t)
        alpha_t = get_alpha(gamma_t)
        alpha_s = get_alpha(gamma_s)
        sigma_t = get_sigma(gamma_t)
        sigma_s = get_sigma(gamma_s)
        predicted_noise = self.model(z, gamma_t)

        mean = alpha_s/alpha_t * (z - sigma_t * c * predicted_noise)
        scale = sigma_s * sqrt(c)
        
        return mean + scale * torch.randn_like(z)
    
    @torch.no_grad()
    def sample(self, batch_size, image_shape, T=250):

        z = torch.randn((batch_size, *image_shape), device=self.device)
        steps = linspace(1.0, 0.0, T + 1, device=self.device)
        
        for i in range(T):
            t = steps[i]
            s = steps[i + 1]
            z = self.ancestral_sampling(z, t, s)
        
        return (z + 1) / 2
    
class VDM(nn.Module):    
    def __init__(self, model, gamma_min, gamma_max, batch_size=128,
                 vocab_size=256, device='cuda', T=250):
        super().__init__()
        self.model = model
        self.vocab_size = vocab_size
        self.device = device
        self.T = T

        self.forward_diffusion = ForwardDiffusion(gamma_min, gamma_max, batch_size, device)
        self.gamma = self.forward_diffusion.gamma
        self.reverse_diffusion = ReverseDiffusion(model, self.gamma, device)
        
    
    def forward(self, batch, noise=None):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, _ = batch
        else:
            x = batch

        x_int = torch.round(x * (self.vocab_size - 1)).long()
        x = 2 * ((x_int + 0.5) / self.vocab_size) - 1

        s, t = self.sample_s_and_t(x.shape[0])
        s.requires_grad_(True)
        t.requires_grad_(True)

        z_t, gamma_t, eps = self.forward_diffusion.sample_z(x, t, noise)
        eps_hat = self.model(z_t, gamma_t)
        loss = self.get_loss(s, t, eps, eps_hat)
        
        return loss.mean(), {"loss": loss.mean()}

    def get_diffusion_loss(self, s, t, eps, eps_hat):
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        weight = torch.expm1(gamma_t - gamma_s)
        mse = ((eps - eps_hat) ** 2).mean(dim=(1, 2, 3))

        return 0.5 * self.T * weight * mse

    def get_prior_loss(self, x):
        gamma_1 = self.gamma(torch.ones(x.size(0), device=self.device))
        gamma_1_expanded = gamma_1.view(-1, *([1] * (x.ndim - 1)))

        alpha_1 = get_alpha(gamma_1_expanded)
        sigma_1 = get_sigma(gamma_1_expanded)

        mu_1 = alpha_1 * x
        var_1 = sigma_1 ** 2

        kl = 0.5 * (var_1 + mu_1**2 - 1 - torch.log(var_1))

        return kl.view(x.size(0), -1).sum(dim=1)

    def get_reconstruction_loss(self, x, z0):
        B, C, H, W = x.shape
        device = self.device

        gamma_0 = self.gamma(torch.zeros(B, device=device))
        gamma_0 = gamma_0.view(-1, 1, 1, 1)

        alpha_0 = get_alpha(gamma_0)
        sigma_0 = get_sigma(gamma_0)

        ks = torch.arange(self.vocab_size, device=device).float()
        ks = ks.view(1, -1, 1, 1, 1)

        xk = 2 * ((ks + 0.5) / self.vocab_size) - 1   
        mu_k = alpha_0 * xk                           
        sigma2 = sigma_0**2                           

        z0_exp = z0.unsqueeze(1)                     

        dist2 = ((z0_exp - mu_k)**2).sum(dim=2)       

        logits = -0.5 * dist2 / sigma2

        x_long = x.long()                             
        x_long = x_long.view(B, -1)                   

        logits_flat = logits.view(B, self.vocab_size, -1)

        log_p = torch.log_softmax(logits_flat, dim=1)
        chosen = log_p.gather(1, x_long.unsqueeze(1)).squeeze(1)

        chosen = chosen.view(B, C, H, W)

        return -chosen.view(B, -1).sum(dim=1)
    
    def sample_s_and_t(self, batch_size):
        i = torch.randint(1, self.T + 1, (batch_size,), device=self.device)
        t = i / self.T
        s = (i - 1) / self.T
        return s, t

class LearnedSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(gamma_min))
        self.w = nn.Parameter(torch.tensor(gamma_max - gamma_min))

    def forward(self, t):
        return self.b + self.w.abs() * t