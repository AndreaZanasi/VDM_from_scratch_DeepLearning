import math
import numpy as np
import torch
from torch import nn


def zero_init(module):
    """Initialize module parameters to zero."""
    for p in module.parameters():
        p.data.zero_()
    return module


class UNet(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        n_blocks=32,
        n_attention_heads=1,
        dropout_prob=0.1,
        norm_groups=32,
        input_channels=3,
        use_fourier_features=True,
        attention_everywhere=False,
        gamma_min=-13.3,
        gamma_max=5.0,
    ):
        super().__init__()

        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.embedding_dim = embedding_dim
        self.use_fourier_features = use_fourier_features

        attention_params = dict(
            n_heads=n_attention_heads,
            n_channels=embedding_dim,
            norm_groups=norm_groups,
        )
        resnet_params = dict(
            ch_in=embedding_dim,
            ch_out=embedding_dim,
            condition_dim=4 * embedding_dim,
            dropout_prob=dropout_prob,
            norm_groups=norm_groups,
        )

        if use_fourier_features:
            self.fourier_features = FourierFeatures(first=7.0, last=8.0, step=1.0)

        # Time embedding MLP
        self.embed_conditioning = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim * 4),
            nn.SiLU(),
        )

        total_input_ch = input_channels
        if use_fourier_features:
            total_input_ch *= 1 + self.fourier_features.num_features

        self.conv_in = nn.Conv2d(total_input_ch, embedding_dim, 3, padding=1)

        # Down path
        self.down_blocks = nn.ModuleList(
            UpDownBlock(
                resnet_block=ResnetBlock(**resnet_params),
                attention_block=AttentionBlock(**attention_params)
                if attention_everywhere
                else None,
            )
            for _ in range(n_blocks)
        )

        # Middle blocks
        self.mid_resnet_block_1 = ResnetBlock(**resnet_params)
        self.mid_attn_block = AttentionBlock(**attention_params)
        self.mid_resnet_block_2 = ResnetBlock(**resnet_params)

        # Up path: input channels doubled by concatenation with skip
        resnet_params_up = dict(resnet_params)
        resnet_params_up["ch_in"] = resnet_params["ch_in"] * 2
        self.up_blocks = nn.ModuleList(
            UpDownBlock(
                resnet_block=ResnetBlock(**resnet_params_up),
                attention_block=AttentionBlock(**attention_params)
                if attention_everywhere
                else None,
            )
            for _ in range(n_blocks + 1)
        )

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=embedding_dim),
            nn.SiLU(),
            zero_init(nn.Conv2d(embedding_dim, input_channels, 3, padding=1)),
        )

    def maybe_concat_fourier(self, z):
        # Ensure 4D
        assert z.dim() == 4, f"Expected 4D input, got {z.shape}"
        if self.use_fourier_features:
            ff = self.fourier_features(z)
            assert ff.dim() == 4, f"FourierFeatures must return 4D, got {ff.shape}"
            return torch.cat([z, ff], dim=1)
        return z

    def forward(self, z, g_t):
        # z must be (B, C, H, W)
        assert z.dim() == 4, f"UNet expects 4D input, got {z.shape}"
        B = z.shape[0]

        # Normalize g_t shape to (B,)
        if isinstance(g_t, (float, int)):
            g_t = torch.tensor([g_t], device=z.device, dtype=z.dtype)
        if g_t.dim() == 0:
            g_t = g_t[None]
        g_t = g_t.view(-1)
        if g_t.shape[0] == 1 and B > 1:
            g_t = g_t.expand(B)
        assert g_t.shape[0] == B, (g_t.shape, B)

        # Rescale gamma to t in [0, 1]
        t = (g_t - self.gamma_min) / (self.gamma_max - self.gamma_min)
        t_embedding = get_timestep_embedding(t, self.embedding_dim)
        cond = self.embed_conditioning(t_embedding)  # (B, 4*embedding_dim)

        h = self.maybe_concat_fourier(z)
        h = self.conv_in(h)  # (B, embedding_dim, H, W)
        hs = []

        # Down path
        for down_block in self.down_blocks:
            hs.append(h)
            h = down_block(h, cond)
        hs.append(h)

        # Middle
        h = self.mid_resnet_block_1(h, cond)
        h = self.mid_attn_block(h)
        h = self.mid_resnet_block_2(h, cond)

        # Up path with skips
        for up_block in self.up_blocks:
            skip = hs.pop()
            h = torch.cat([h, skip], dim=1)
            h = up_block(h, cond)

        prediction = self.conv_out(h)
        assert prediction.shape == z.shape, (prediction.shape, z.shape)
        return prediction


def get_timestep_embedding(
    timesteps,
    embedding_dim: int,
    dtype=torch.float32,
    max_timescale=10_000,
    min_timescale=1,
):
    """Positional/timestep embedding as in DDPM/VDM.[web:2][web:39]"""
    assert timesteps.ndim == 1
    assert embedding_dim % 2 == 0
    timesteps = timesteps.to(dtype) * 1000.0  # map [0,1] to [0,1000]

    num_timescales = embedding_dim // 2
    inv_timescales = torch.logspace(
        -np.log10(min_timescale),
        -np.log10(max_timescale),
        num_timescales,
        device=timesteps.device,
        dtype=dtype,
    )
    emb = timesteps[:, None] * inv_timescales[None, :]  # (B, D/2)
    return torch.cat([emb.sin(), emb.cos()], dim=1)  # (B, D)


class FourierFeatures(nn.Module):
    def __init__(self, first=5.0, last=6.0, step=1.0):
        super().__init__()
        self.register_buffer(
            "freqs_exponent",
            torch.arange(first, last + 1e-8, step),
            persistent=False,
        )

    @property
    def num_features(self):
        return len(self.freqs_exponent) * 2

    def forward(self, x):
        # x: (B, C, H, W)
        assert x.dim() == 4, f"Expected 4D input, got {x.shape}"
        B, C, H, W = x.shape

        freqs_exponent = self.freqs_exponent.to(dtype=x.dtype, device=x.device)
        freqs = 2.0 ** freqs_exponent * 2 * math.pi  # (F,)
        freqs = freqs.view(1, -1, 1, 1, 1)          # (1, F, 1, 1, 1)

        x_expanded = x.unsqueeze(1)                 # (B, 1, C, H, W)
        features = freqs * x_expanded               # (B, F, C, H, W)
        features = features.reshape(B, -1, H, W)    # (B, F*C, H, W)

        return torch.cat([features.sin(), features.cos()], dim=1)  # (B, 2*F*C, H, W)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out=None,
        condition_dim=None,
        dropout_prob=0.0,
        norm_groups=32,
    ):
        super().__init__()
        ch_out = ch_in if ch_out is None else ch_out
        self.ch_out = ch_out
        self.condition_dim = condition_dim

        self.net1 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=ch_in),
            nn.SiLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
        )

        if condition_dim is not None:
            self.cond_proj = zero_init(nn.Linear(condition_dim, ch_out, bias=False))
        else:
            self.cond_proj = None

        layers2 = [
            nn.GroupNorm(num_groups=norm_groups, num_channels=ch_out),
            nn.SiLU(),
        ]
        if dropout_prob > 0.0:
            layers2.append(nn.Dropout(dropout_prob))
        layers2.append(zero_init(nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)))
        self.net2 = nn.Sequential(*layers2)

        if ch_in != ch_out:
            self.skip_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)
        else:
            self.skip_conv = None

    def forward(self, x, condition):
        h = self.net1(x)
        if self.cond_proj is not None and condition is not None:
            assert condition.shape == (x.shape[0], self.condition_dim)
            cond = self.cond_proj(condition)
            cond = cond[:, :, None, None]
            h = h + cond
        h = self.net2(h)
        if self.skip_conv is not None:
            x = self.skip_conv(x)
        assert x.shape == h.shape
        return x + h


class Attention(nn.Module):
    """Self-attention with heads in channel dimension."""

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        assert qkv.dim() >= 3, qkv.dim()
        assert qkv.shape[1] % (3 * self.n_heads) == 0
        spatial_dims = qkv.shape[2:]
        # (B, 3*C, T)
        qkv = qkv.reshape(*qkv.shape[:2], -1)
        out = attention_inner_heads(qkv, self.n_heads)  # (B, C, T)
        return out.reshape(*out.shape[:2], *spatial_dims)


def attention_inner_heads(qkv, num_heads):
    """Computes attention with heads inside of qkv in the channel dimension."""
    bs, width, length = qkv.shape
    ch = width // (3 * num_heads)

    # Split into (q, k, v) of shape (B, H*C, T).
    q, k, v = qkv.chunk(3, dim=1)

    # Rescale q and k.
    scale = ch ** (-0.25)
    q = q * scale
    k = k * scale

    # Reshape qkv to (B*H, C, T).
    new_shape = (bs * num_heads, ch, length)
    q = q.reshape(*new_shape)
    k = k.reshape(*new_shape)
    v = v.reshape(*new_shape)

    # Compute attention.
    weight = torch.einsum("bct,bcs->bts", q, k)  # (B*H, T, T)
    weight = torch.softmax(weight.float(), dim=-1).to(weight.dtype)
    out = torch.einsum("bts,bcs->bct", weight, v)  # (B*H, C, T)
    return out.reshape(bs, num_heads * ch, length)  # (B, H*C, T)


class AttentionBlock(nn.Module):
    def __init__(self, n_heads, n_channels, norm_groups):
        super().__init__()
        assert n_channels % n_heads == 0
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=n_channels),
            nn.Conv2d(n_channels, 3 * n_channels, kernel_size=1),
            Attention(n_heads),
            zero_init(nn.Conv2d(n_channels, n_channels, kernel_size=1)),
        )

    def forward(self, x):
        return self.layers(x) + x


class UpDownBlock(nn.Module):
    def __init__(self, resnet_block, attention_block=None):
        super().__init__()
        self.resnet_block = resnet_block
        self.attention_block = attention_block

    def forward(self, x, cond):
        x = self.resnet_block(x, cond)
        if self.attention_block is not None:
            x = self.attention_block(x)
        return x
