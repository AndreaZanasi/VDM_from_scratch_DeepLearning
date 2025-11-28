import math
import numpy as np
import torch
from torch import nn

def zero_init(module):
    """
    Initialize all parameters of a module to zero.
    Args:
        module: PyTorch module to initialize
    Returns:
        module: The same module with zero-initialized parameters
    """
    for p in module.parameters():
        p.data.zero_()
    return module

def get_timestep_embedding(
    timesteps,
    embedding_dim,
    dtype=torch.float32,
    max_timescale=10_000,
    min_timescale=1,
):
    """
    Create sinusoidal timestep embeddings. Converts scalar timesteps to high-dimensional 
    embeddings using sinusoidal functions of different frequencies. 
    Args:
        timesteps: (B,) normalized timesteps in [0, 1]
        embedding_dim: dimension of the embedding
        dtype: torch dtype for the embedding
        max_timescale: maximum period of sinusoidal functions
        min_timescale: minimum period of sinusoidal functions
    Returns:
        emb: (B, embedding_dim) timestep embeddings
    """
    assert timesteps.ndim == 1, "Timesteps must be 1D"
    assert embedding_dim % 2 == 0, "Embedding dimension must be even"

    timesteps = timesteps.to(dtype) * 1000.0
    num_timescales = embedding_dim // 2
    inv_timescales = torch.logspace(
        -np.log10(min_timescale),
        -np.log10(max_timescale),
        num_timescales,
        device=timesteps.device,
        dtype=dtype,
    )
    emb = timesteps[:, None] * inv_timescales[None, :]

    return torch.cat([emb.sin(), emb.cos()], dim=1)

class FourierFeatures(nn.Module):
    """
    Fourier features for capturing fine-scale details in the data.
    Applies sinusoidal transformations at different frequencies to the input,
    allowing the network to better model high-frequency details.
    Args:
        first: float, first exponent (minimum frequency: 2^first)
        last: float, last exponent (maximum frequency: 2^last)
        step: float, step size between exponents
    """
    def __init__(self, first=5.0, last=6.0, step=1.0):
        super().__init__()
        self.register_buffer(
            "freqs_exponent",
            torch.arange(first, last + 1e-8, step),
            persistent=False,
        )

    @property
    def num_features(self):
        """
        Total number of Fourier feature channels.
        Returns 2 features (sin and cos) for each frequency.
        Returns:
            int: number of Fourier feature channels
        """
        return len(self.freqs_exponent) * 2

    def forward(self, x):
        """
        Compute Fourier features for input data.
        Args:
            x: (B, C, H, W) input data
        Returns:
            features: (B, 2*F*C, H, W) Fourier features where F is number of frequencies
        """
        assert x.dim() == 4, f"Expected 4D input, got {x.shape}"
        B, C, H, W = x.shape

        freqs_exponent = self.freqs_exponent.to(dtype=x.dtype, device=x.device)
        freqs = 2.0 ** freqs_exponent * 2 * math.pi
        freqs = freqs.view(1, -1, 1, 1, 1)

        x_expanded = x.unsqueeze(1)

        features = freqs * x_expanded
        features = features.reshape(B, -1, H, W)

        return torch.cat([features.sin(), features.cos()], dim=1)

class ResnetBlock(nn.Module):
    """
    Residual block with conditioning and skip connection.
    Implements a residual block that processes spatial features while being
    conditioned on time embeddings.
    Args:
        ch_in: int, number of input channels
        ch_out: int, number of output channels (defaults to ch_in)
        condition_dim: int, dimension of conditioning vector (e.g., time embedding)
        dropout_prob: float, dropout probability
        norm_groups: int, number of groups for GroupNorm
    """

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
        """
        Forward pass with optional time conditioning.
        Args:
            x: (B, ch_in, H, W) input features
            condition: (B, condition_dim) conditioning vector
        Returns:
            out: (B, ch_out, H, W) output features
        """
        h = self.net1(x)

        if self.cond_proj is not None and condition is not None:
            assert condition.shape == (x.shape[0], self.condition_dim)
            cond = self.cond_proj(condition)
            cond = cond[:, :, None, None]
            h = h + cond

        h = self.net2(h)
        if self.skip_conv is not None:
            x = self.skip_conv(x)

        assert x.shape == h.shape, f"Shape mismatch: {x.shape} vs {h.shape}"
        return x + h

class Attention(nn.Module):
    """
    Multi-head self-attention mechanism.
    Implements self-attention with multiple heads processed in the channel dimension.
    The attention mechanism allows the network to capture long-range dependencies
    in the spatial dimensions.
    Args:
        n_heads: int, number of attention heads
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
    
    def attention_inner_heads(self, qkv):
        """
        Compute scaled dot-product attention with multiple heads.
        Implements the attention mechanism: Attention(Q, K, V) = softmax(QK^T / sqrt(d))V
        with multiple heads processed in parallel.
        Args:
            qkv: (B, 3*H*C, T) concatenated queries, keys, and values
        Returns:
            out: (B, H*C, T) attention output
        """
        bs, width, length = qkv.shape
        ch = width // (3 * self.n_heads)

        q, k, v = qkv.chunk(3, dim=1)

        scale = ch ** (-0.25)
        q = q * scale
        k = k * scale

        new_shape = (bs * self.n_heads, ch, length)
        q = q.reshape(*new_shape)
        k = k.reshape(*new_shape)
        v = v.reshape(*new_shape)

        weight = torch.einsum("bct,bcs->bts", q, k)
        weight = torch.softmax(weight.float(), dim=-1).to(weight.dtype)
        out = torch.einsum("bts,bcs->bct", weight, v)

        return out.reshape(bs, self.n_heads * ch, length)

    def forward(self, qkv):
        """
        Compute multi-head self-attention.
        Args:
            qkv: (B, 3*n_heads*C, H, W) concatenated queries, keys, and values
        Returns:
            out: (B, n_heads*C, H, W) attention output
        """
        assert qkv.dim() >= 3, f"Expected at least 3D input, got {qkv.dim()}D"
        assert qkv.shape[1] % (3 * self.n_heads) == 0, "Channel dimension must be divisible by 3*n_heads"

        spatial_dims = qkv.shape[2:]
        qkv = qkv.reshape(*qkv.shape[:2], -1)
        out = self.attention_inner_heads(qkv)

        return out.reshape(*out.shape[:2], *spatial_dims)

class AttentionBlock(nn.Module):
    """
    Self-attention block with residual connection.
    Wraps the attention mechanism with normalization and a residual connection,
    allowing the network to optionally use attention while maintaining gradient flow.
    Args:
        n_heads: int, number of attention heads
        n_channels: int, total number of channels (must be divisible by n_heads)
        norm_groups: int, number of groups for GroupNorm
    """

    def __init__(self, n_heads, n_channels, norm_groups):
        super().__init__()
        assert n_channels % n_heads == 0, "n_channels must be divisible by n_heads"

        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=n_channels),
            nn.Conv2d(n_channels, 3 * n_channels, kernel_size=1),
            Attention(n_heads),
            zero_init(nn.Conv2d(n_channels, n_channels, kernel_size=1)),
        )

    def forward(self, x):
        """
        Apply self-attention with residual connection.
        Args:
            x: (B, n_channels, H, W) input features
        Returns:
            out: (B, n_channels, H, W) output features
        """
        return self.layers(x) + x

class UpDownBlock(nn.Module):
    """
    Combined ResNet and optional Attention block.
    A building block that combines a residual block with optional self-attention.
    Used in both the downsampling and upsampling paths of the UNet.
    Args:
        resnet_block: ResnetBlock module
        attention_block: AttentionBlock module or None
    """

    def __init__(self, resnet_block, attention_block=None):
        super().__init__()
        self.resnet_block = resnet_block
        self.attention_block = attention_block

    def forward(self, x, cond):
        """
        Forward pass through ResNet block and optional attention.
        Args:
            x: (B, C, H, W) input features
            cond: (B, condition_dim) conditioning vector
        Returns:
            out: (B, C, H, W) output features
        """
        x = self.resnet_block(x, cond)
        if self.attention_block is not None:
            x = self.attention_block(x)
        return x

class UNet(nn.Module):
    """
    UNet architecture for noise prediction.
    This UNet predicts the noise that was added to clean data x to produce
    the noisy latent z_t. The network is conditioned on the noise level gamma_t
    and optionally enhanced with Fourier features for better high-frequency modeling.
    Architecture:
    - Input processing with optional Fourier features
    - Downsampling path with ResNet blocks and optional attention
    - Middle blocks with ResNet and attention
    - Upsampling path with skip connections from downsampling path
    - Output projection
    Args:
        embedding_dim: int, base channel dimension
        n_blocks: int, number of blocks in down/up paths
        n_attention_heads: int, number of attention heads
        dropout_prob: float, dropout probability in ResNet blocks
        norm_groups: int, number of groups for GroupNorm
        input_channels: int, number of input data channels (3 for RGB)
        use_fourier_features: bool, whether to use Fourier features
        attention_everywhere: bool, whether to use attention in all blocks
        gamma_min: float, minimum value of gamma_t (most noisy)
        gamma_max: float, maximum value of gamma_t (least noisy)
    """

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
            self.fourier_features = FourierFeatures(first=2.0, last=5.0, step=1.0) #changed to capture more high frequency details

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

        self.down_blocks = nn.ModuleList(
            [
                UpDownBlock(
                    resnet_block=ResnetBlock(**resnet_params),
                    attention_block=AttentionBlock(**attention_params)
                    if attention_everywhere
                    else None,
                )
                for _ in range(n_blocks)
            ]
        )

        self.mid_resnet_block_1 = ResnetBlock(**resnet_params)
        self.mid_attn_block = AttentionBlock(**attention_params)
        self.mid_resnet_block_2 = ResnetBlock(**resnet_params)

        resnet_params_up = dict(resnet_params)
        resnet_params_up["ch_in"] = resnet_params["ch_in"] * 2
        self.up_blocks = nn.ModuleList(
            [
                UpDownBlock(
                    resnet_block=ResnetBlock(**resnet_params_up),
                    attention_block=AttentionBlock(**attention_params)
                    if attention_everywhere
                    else None,
                )
                for _ in range(n_blocks + 1)
            ]
        )

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=embedding_dim),
            nn.SiLU(),
            zero_init(nn.Conv2d(embedding_dim, input_channels, 3, padding=1)),
        )

    def concat_fourier(self, z):
        """
        Optionally concatenate Fourier features to input.
        If Fourier features are enabled, computes sinusoidal features at different
        frequencies and concatenates them to the input. This helps the network
        model high-frequency details important for likelihood estimation.
        Args:
            z: (B, C, H, W) input data
        Returns:
            z_aug: (B, C*(1+2*F), H, W) input with optional Fourier features
        """
        assert z.dim() == 4, f"Expected 4D input, got {z.shape}"

        if self.use_fourier_features:
            ff = self.fourier_features(z)
            assert ff.dim() == 4, f"FourierFeatures must return 4D, got {ff.shape}"
            return torch.cat([z, ff], dim=1)

        return z

    def forward(self, z, g_t):
        """
        Predict noise eps given noisy data z_t and noise level gamma_t.
        The forward pass:
        1. Normalize gamma_t to timestep t ∈ [0, 1]
        2. Create time embedding from t
        3. Optionally add Fourier features to z_t
        4. Process through UNet with skip connections
        5. Output noise prediction eps(z_t; gamma_t)
        Args:
            z: (B, C, H, W) noisy data z_t
            g_t: (B,) or scalar, noise level gamma_t = log(SNR(t))
        Returns:
            eps_pred: (B, C, H, W) predicted noise eps(z_t; gamma_t)
        """
        assert z.dim() == 4, f"UNet expects 4D input, got {z.shape}"
        B = z.shape[0]

        if isinstance(g_t, (float, int)):
            g_t = torch.tensor([g_t], device=z.device, dtype=z.dtype)
        if g_t.dim() == 0:
            g_t = g_t[None]
        g_t = g_t.view(-1)
        if g_t.shape[0] == 1 and B > 1:
            g_t = g_t.expand(B)
        assert g_t.shape[0] == B, f"Batch size mismatch: γ_t has {g_t.shape[0]}, z has {B}"

        t = (g_t - self.gamma_min) / (self.gamma_max - self.gamma_min)
        t_embedding = get_timestep_embedding(t, self.embedding_dim)
        cond = self.embed_conditioning(t_embedding)

        h = self.concat_fourier(z)
        h = self.conv_in(h)

        hs = []
        for down_block in self.down_blocks:
            hs.append(h)
            h = down_block(h, cond)
        hs.append(h)

        h = self.mid_resnet_block_1(h, cond)
        h = self.mid_attn_block(h)
        h = self.mid_resnet_block_2(h, cond)

        for up_block in self.up_blocks:
            skip = hs.pop()
            h = torch.cat([h, skip], dim=1)
            h = up_block(h, cond)

        prediction = self.conv_out(h)
        assert prediction.shape == z.shape, f"Output shape {prediction.shape} != input shape {z.shape}"
        return prediction
