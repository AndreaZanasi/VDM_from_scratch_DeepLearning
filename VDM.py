import torch
import numpy as np
from torch import nn, sigmoid, sqrt, exp
from torch.special import expm1
from tqdm import tqdm

def get_alpha(gamma):
    return sqrt(sigmoid(-gamma))


def get_sigma(gamma):
    return sqrt(sigmoid(gamma))


def get_snr(gamma):
    return exp(-gamma)


class FixedLinearSchedule(nn.Module):
    def __init__(self, gamma_min=-13.3, gamma_max=5.0):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, t):
        # t in [0, 1]
        return self.gamma_min + t * (self.gamma_max - self.gamma_min)


class LearnedSchedule(nn.Module):
    def __init__(self, gamma_min=-13.3, gamma_max=5.0):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        self.l1 = nn.Linear(1, 1, bias=True)
        self.l2 = nn.Linear(1, 1024, bias=False)
        self.l3 = nn.Linear(1024, 1, bias=False)

        # Initialize weights non-negative to encourage monotonicity
        with torch.no_grad():
            self.l1.weight.abs_()
            self.l2.weight.abs_()
            self.l3.weight.abs_()

    def forward(self, t):
        # t: (B,) or (B,1) in [0,1]
        t = t.view(-1, 1)

        gamma_tilde = self.forward_tilde(t)

        gamma_tilde_0 = self.forward_tilde(torch.zeros_like(t))
        gamma_tilde_1 = self.forward_tilde(torch.ones_like(t))

        # Rescale to [gamma_min, gamma_max]
        gamma_t = self.gamma_min + (self.gamma_max - self.gamma_min) * (
            (gamma_tilde - gamma_tilde_0) / (gamma_tilde_1 - gamma_tilde_0)
        )

        return gamma_t.squeeze(-1)

    def forward_tilde(self, t):
        # Monotone scalar MLP with clamped weights
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
        """
        x: (B, C, H, W) in [-1, 1]
        t: (B,) in [0,1]
        """
        gamma_t = self.gamma(t)  # (B,)
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
        # image_shape: (C, H, W)
        z = torch.randn((batch_size, *image_shape), device=device)  # (B, C, H, W)
        steps = torch.linspace(1.0, 0.0, T + 1, device=device)

        for i in tqdm(range(T), desc='Reverse Diffusion Sampling'):
            t = steps[i].expand(batch_size)      # (B,)
            s = steps[i + 1].expand(batch_size)  # (B,)

            gamma_t = self.gamma(t)              # (B,)
            gamma_s = self.gamma(s)              # (B,)

            # pad to (B, 1, 1, 1) to match z: (B, C, H, W)
            gamma_t_padded = gamma_t.view(-1, *([1] * (z.ndim - 1)))  # (B,1,1,1)
            gamma_s_padded = gamma_s.view(-1, *([1] * (z.ndim - 1)))  # (B,1,1,1)

            alpha_t = get_alpha(gamma_t_padded)
            alpha_s = get_alpha(gamma_s_padded)
            sigma_t = get_sigma(gamma_t_padded)
            sigma_s = get_sigma(gamma_s_padded)

            eps_hat = self.model(z, gamma_t)     # UNet sees (B, C, H, W) and (B,)

            c = -expm1(gamma_s_padded - gamma_t_padded)
            mean = (alpha_s / alpha_t) * (z - sigma_t * c * eps_hat)

            if i < T - 1:
                noise = torch.randn_like(z)
                z = mean + sigma_s * torch.sqrt(c) * noise
            else:
                z = mean

        z = torch.clamp((z + 1) / 2, 0, 1)
        return z


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
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        if learned_schedule:
            self.gamma = LearnedSchedule(gamma_min, gamma_max)
        else:
            self.gamma = FixedLinearSchedule(gamma_min, gamma_max)

        self.forward_diffusion = ForwardDiffusion(self.gamma)
        self.reverse_diffusion = ReverseDiffusion(model, self.gamma)

    def forward(self, batch):
        # unpack (x, y) if needed
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, _ = batch
        else:
            x = batch

        x = x.to(self.device)
        B = x.shape[0]

        # x in [0,1] -> discrete 0..vocab_size-1 -> continuous in [-1,1]
        x_int = torch.round(x * (self.vocab_size - 1)).long()
        x_cont = 2 * ((x_int + 0.5) / self.vocab_size) - 1  # in [-1,1]

        # pick discrete step i in {1,..,T} and corresponding times
        i = torch.randint(1, self.T + 1, (B,), device=self.device)
        t = i.float() / self.T           # current time
        s = (i - 1).float() / self.T     # previous time (not used in new loss)

        # forward diffusion
        z_t, gamma_t, eps = self.forward_diffusion.sample_z(x_cont, t)
        eps_hat = self.model(z_t, gamma_t)

        # -------- stable diffusion loss (discrete-time approx of continuous-time VDM) --------
        # For linear gamma(t), dgamma/dt is constant:
        dgamma_dt = (self.gamma_max - self.gamma_min)  # scalar
        mse = ((eps - eps_hat) ** 2).mean(dim=(1, 2, 3))
        diffusion_loss = 0.5 * dgamma_dt * mse  # shape (B,)

        # -------- prior loss --------
        gamma_1 = self.gamma(torch.ones(B, device=self.device))
        snr_1 = get_snr(gamma_1)
        prior_factor = (snr_1 / (snr_1 + 1)).view(B, 1, 1, 1)
        prior_loss = (prior_factor * (x_cont ** 2)).mean(dim=(1, 2, 3))

        # -------- discrete reconstruction term p(x|z_0) --------
        t_0 = torch.zeros(B, device=self.device)
        z_0, gamma_0, _ = self.forward_diffusion.sample_z(x_cont, t_0)

        gamma_0_padded = gamma_0.view(B, 1, 1, 1)
        alpha_0 = get_alpha(gamma_0_padded)
        sigma_0 = get_sigma(gamma_0_padded)

        x_vals = 2 * (
            (torch.arange(self.vocab_size, device=self.device).float() + 0.5)
            / self.vocab_size
        ) - 1  # (V,)
        mu_vals = alpha_0.unsqueeze(-1) * x_vals.view(1, 1, 1, 1, -1)

        z_0_exp = z_0.unsqueeze(-1)
        dist_sq = ((z_0_exp - mu_vals) ** 2) / (sigma_0.unsqueeze(-1) ** 2)
        logits = -0.5 * dist_sq

        log_probs = torch.log_softmax(logits, dim=-1)
        x_int_exp = x_int.unsqueeze(-1)
        reconstruction_loss = -log_probs.gather(-1, x_int_exp).squeeze(-1).mean(
            dim=(1, 2, 3)
        )

        # -------- total loss --------
        loss = diffusion_loss + prior_loss + reconstruction_loss  # (B,)
        loss_mean = loss.mean()

        stats = {
            "loss": loss_mean.item(),
            "diffusion": diffusion_loss.mean().item(),
            "prior": prior_loss.mean().item(),
            "reconstruction": reconstruction_loss.mean().item(),
        }

        return loss_mean, stats
