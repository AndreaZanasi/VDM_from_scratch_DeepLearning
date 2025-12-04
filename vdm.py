import torch
from torch import nn, sigmoid, sqrt, exp, log_softmax
from torch.special import expm1
from tqdm import tqdm
from noise_scheduler import LinearSchedule, LearnedSchedule
import numpy as np

def get_alpha(gamma):
    return sqrt(sigmoid(-gamma))

def get_sigma(gamma):
    return sqrt(sigmoid(gamma))

def get_snr(gamma):
    return exp(-gamma)

class ForwardDiffusion(nn.Module):
    def __init__(self, gamma_schedule):
        super().__init__()
        self.gamma = gamma_schedule

    def forward(self, x, t, eps=None):
        """
        Sample z_t given x and timestep t, we get a noisy version of x at time t.
        Args:
            x: (B, C, H, W) input data in [-1, 1] : clean data
            t: (B,) timesteps in [0, 1] : normalized time
            eps: (B, C, H, W) optional noise to use, if None sampled from N(0, I) : noise
        Returns:
            z_t: (B, C, H, W) noisy data at time t
            gamma_t: (B,) gamma values at time t
            eps: (B, C, H, W) noise used
        """
        gamma_t = self.gamma(t)
        gamma_t_padded = gamma_t.view(-1, *([1] * (x.ndim - 1)))
        
        alpha_t = get_alpha(gamma_t_padded)
        sigma_t = get_sigma(gamma_t_padded)
        
        if eps is None:
            eps = torch.randn_like(x)
        
        z_t = alpha_t * x + sigma_t * eps
        
        return z_t, gamma_t, eps

class ReverseDiffusion(nn.Module):
    def __init__(self, model, gamma):
        super().__init__()
        self.model = model
        self.gamma = gamma

    @torch.no_grad()
    def sample(self, batch_size, shape, T=1000, device='cuda'):
        """
        Sample from the reverse diffusion process.
        Args:
            batch_size: int, number of samples to generate
            shape: tuple, shape of each sample (C, H, W)
            T: int, number of diffusion steps
            device: str, device to perform computation on
        Returns:
            z: (B, C, H, W) generated samples in [0, 1]
        """
        z = torch.randn(batch_size, *shape, device=device)
        timesteps = torch.linspace(1.0, 0.0, T + 1, device=device)

        for i in tqdm(range(T), desc='Sampling'):
            t = timesteps[i].expand(batch_size)
            s = timesteps[i + 1].expand(batch_size)
            is_last_step = (i == T - 1)

            z = self.denoise_step(z, t, s, is_last_step)

        z = torch.clamp((z + 1) / 2, 0, 1)
        return z

    def denoise_step(self, z_t, t, s, last_step):
        """
        Perform a single denoising step from z_t at time t to z_s at time s.
        Args:
            z_t: (B, C, H, W) noisy data at time t
            t: (B,) current timesteps
            s: (B,) next timesteps
            last_step: bool, whether this is the last denoising step
        Returns:
            z_s: (B, C, H, W) denoised data at time s
        """
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        
        gamma_t_expanded = gamma_t.view(-1, *([1] * (z_t.ndim - 1)))
        gamma_s_expanded = gamma_s.view(-1, *([1] * (z_t.ndim - 1)))
        
        alpha_t = get_alpha(gamma_t_expanded)
        alpha_s = get_alpha(gamma_s_expanded)
        sigma_t = get_sigma(gamma_t_expanded)
        sigma_s = get_sigma(gamma_s_expanded)
        
        eps_hat = self.model(z_t, gamma_t)
        
        c = -expm1(gamma_s_expanded - gamma_t_expanded)
        mean = (alpha_s / alpha_t) * (z_t - sigma_t * c * eps_hat)
        
        if not last_step:
            noise = torch.randn_like(z_t)
            z_s = mean + sigma_s * torch.sqrt(c) * noise
        else:
            z_s = mean
        
        return z_s
    
    @torch.no_grad()
    def sample_from_noisy(self, z_t, t_start, T, device='cuda'):
        """
        Reverse diffusion starting from a given noisy z_t down to t=0
        Args:
            z_t: noisy image at t_start
            t_start: fraction of total T where z_t is located (0-1)
            T: number of denoising steps from t_start to t=0
            device: device
        Returns:
            z_0: reconstructed image
        """
        batch_size = z_t.shape[0]
        timesteps = torch.linspace(t_start, 0.0, T + 1, device=device)

        z = z_t.clone()
        for i in range(T):
            t = timesteps[i].expand(batch_size)
            s = timesteps[i + 1].expand(batch_size)
            is_last_step = (i == T - 1)
            z = self.denoise_step(z, t, s, is_last_step)
        return z


class DiffusionLoss(nn.Module):
    def __init__(self, gamma, T):
        super().__init__()
        self.gamma = gamma
        self.T = T

    def forward(self, eps, eps_hat, t):
        """
        Compute the diffusion loss between true noise and predicted noise.
        It measures how well the model predicts the noise added during the forward diffusion process.
        Args:
            eps: (B, C, H, W) true noise added
            eps_hat: (B, C, H, W) predicted noise by the model
        Returns:
            loss: (B,) diffusion loss for each sample in the batch
        """
        s = t - (1.0 / self.T)
        s = torch.clamp(s, min=0.0)

        gamma_t = self.gamma(t).view(-1, 1, 1, 1)
        gamma_s = self.gamma(s).view(-1, 1, 1, 1)
        
        weight = expm1(gamma_t - gamma_s) 
        
        mse = (eps - eps_hat) ** 2
        loss = 0.5 * self.T * weight * mse
        
        return loss.sum(dim=(1, 2, 3))

class PriorLoss(nn.Module):
    def __init__(self, gamma, device):
        super().__init__()
        self.gamma = gamma
        self.device = device

    def forward(self, x, batch_size):
        """
        Compute the prior loss, it measures how well the model's prior matches the data distribution.
        Args:
            x: (B, C, H, W) input data in [-1, 1]
            batch_size: int, number of samples in the batch
        Returns:
            loss: (B,) prior loss for each sample in the batch
        """
        gamma_1 = self.gamma(torch.ones(batch_size, device=self.device))
        gamma_1 = gamma_1.view(batch_size, 1, 1, 1)
        sigma2_1 = sigmoid(gamma_1)
        alpha2_1 = sigmoid(-gamma_1)
        
        
        mean_sq = alpha2_1 * (x ** 2)
        
        log_sigma2_1 = -torch.nn.functional.softplus(-gamma_1)
        
        kl = 0.5 * (-log_sigma2_1 + sigma2_1 + mean_sq - 1)
        
        return kl.sum(dim=(1, 2, 3))

class ReconstructionLoss(nn.Module):
    def __init__(self, forward_diffusion, vocab_size, device):
        super().__init__()
        self.forward_diffusion = forward_diffusion
        self.vocab_size = vocab_size
        self.device = device

    def forward(self, x_int, x_cont, batch_size):
        """
        Compute the reconstruction loss, measuring how well the model can reconstruct
        the original discrete data from the noisy continuous data.
        Args:
            x_int: (B, C, H, W) discrete input data as integers
            x_cont: (B, C, H, W) continuous input data in [-1, 1]
            batch_size: int, number of samples in the batch
        Returns:
            loss: (B,) reconstruction loss for each sample in the batch
        """
        t_0 = torch.zeros(batch_size, device=self.device)
        z_0, gamma_0, _ = self.forward_diffusion(x_cont, t_0)
        
        gamma_0 = gamma_0.view(batch_size, 1, 1, 1)
        alpha_0 = get_alpha(gamma_0)
        sigma_0 = get_sigma(gamma_0)
        
        x_vals = self.get_discrete_values()
        mu_vals = alpha_0.unsqueeze(-1) * x_vals.view(1, 1, 1, 1, -1)
        
        logits = self.compute_logits(z_0, mu_vals, sigma_0)
        log_probs = log_softmax(logits, dim=-1)
        
        x_int_padded = x_int.unsqueeze(-1)
        nll = -log_probs.gather(-1, x_int_padded).squeeze(-1)
        
        return nll.sum(dim=(1, 2, 3))

    def get_discrete_values(self):
        """
        Get the discrete values in [-1, 1] corresponding to the vocabulary size.
        It creates a tensor of shape (vocab_size,) where each value represents
        the center of each discrete bin in the continuous space.
        Returns:
            bins: (vocab_size,) discrete values in [-1, 1]
        """
        bins = torch.arange(self.vocab_size, device=self.device).float()
        return 2 * ((bins + 0.5) / self.vocab_size) - 1

    def compute_logits(self, z_0, mu_vals, sigma_0):
        """
        Compute the logits for the reconstruction loss.
        It calculates the negative squared distance between the noisy data z_0
        and the means mu_vals, scaled by the variance sigma_0.
        Args:
            z_0: (B, C, H, W) noisy data at time 0
            mu_vals: (1, 1, 1, 1, V) means for each discrete value
            sigma_0: (B, 1, 1, 1) standard deviation at time 0
        Returns:
            logits: (B, C, H, W, V) logits for each discrete value
        """
        squared_dist = ((z_0.unsqueeze(-1) - mu_vals) ** 2) / (sigma_0.unsqueeze(-1) ** 2)
        return -0.5 * squared_dist

class VDM(nn.Module):
    def __init__(
        self,
        model,
        gamma_min=-13.3,
        gamma_max=5.0,
        vocab_size=256,
        T=1000,
        device='cuda',
        learned_schedule=False,
    ):
        super().__init__()
        self.model = model
        self.vocab_size = vocab_size
        self.device = device
        self.T = T
        
        schedule = LearnedSchedule if learned_schedule else LinearSchedule
        self.gamma = schedule(gamma_min, gamma_max)
        
        self.forward_diffusion = ForwardDiffusion(self.gamma)
        self.reverse_diffusion = ReverseDiffusion(self.model, self.gamma)
        
        self.diffusion_loss = DiffusionLoss(self.gamma, T)
        self.prior_loss = PriorLoss(self.gamma, device)
        self.reconstruction_loss = ReconstructionLoss(self.forward_diffusion, vocab_size, device)

    def forward(self, batch):
        """
        Compute the total loss for a given batch.
        First, it extracts the data from the batch, discretizes it,
        converts it to continuous form, samples timesteps, get noise from
        the forward diffusion, get the model's noise prediction, and computes
        the diffusion, prior, and reconstruction losses.
        then, it sums these losses to get the total loss.
        Args:
            batch: input batch, can be (data, labels) or just data
        Returns:
            total_loss: scalar tensor, total loss for the batch
            stats: dict, individual loss components for logging
        """
        x = self.extract_data(batch).to(self.device)
        batch_size = x.shape[0]
        
        x_int = self.discretize(x)
        x_cont = self.to_continuous(x_int)
        
        t = self.sample_t(batch_size)
        
        z_t, gamma_t, eps = self.forward_diffusion(x_cont, t)
        eps_hat = self.model(z_t, gamma_t)

        bpd_factor = 1 / (np.prod(x.shape[1:]) * np.log(2))
        
        diffusion_loss = self.diffusion_loss(eps, eps_hat, t) * bpd_factor
        prior_loss = self.prior_loss(x_cont, batch_size) * bpd_factor
        reconstruction_loss = self.reconstruction_loss(x_int, x_cont, batch_size) * bpd_factor
        
        total_loss = diffusion_loss + prior_loss + reconstruction_loss
        
        stats = {
            "loss": total_loss.mean().item(),
            "diffusion": diffusion_loss.mean().item(),
            "prior": prior_loss.mean().item(),
            "reconstruction": reconstruction_loss.mean().item(),
        }
        
        return total_loss.mean(), stats

    def extract_data(self, batch):
        """
        Extract data from the batch, handling cases where batch is a tuple (data, labels).
        Args:
            batch: input batch, can be (data, labels) or just data
        Returns:
            data: extracted data tensor
        """
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0]
        return batch

    def discretize(self, x):
        """
        Discretize continuous data in [-1, 1] to integer values in {0, ..., vocab_size-1}.
        Args:
            x: (B, C, H, W) continuous input data in [-1, 1]
        Returns:
            x_int: (B, C, H, W) discrete input data as integers
        """
        return torch.round(x * (self.vocab_size - 1)).long()

    def to_continuous(self, x_int):
        """
        Convert discrete integer data back to continuous data in [-1, 1].
        Args:
            x_int: (B, C, H, W) discrete input data as integers
        Returns:
            x: (B, C, H, W) continuous data in [-1, 1]
        """
        return 2 * ((x_int + 0.5) / self.vocab_size) - 1

    def sample_t(self, batch_size):
        """
        Sample timesteps t in [0, 1] using low-discrepancy sampler.
        """
        u0 = torch.rand(1, device=self.device)  # single random offset
        t = (u0 + torch.arange(batch_size, device=self.device).float() / batch_size) % 1.0
        return t
