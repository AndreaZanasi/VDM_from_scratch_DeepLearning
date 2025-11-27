# References
- Repo reference: https://github.com/addtt/variational-diffusion-models
- Paper reference: https://arxiv.org/pdf/2107.00630

# Forward Diffusion Process

### Goal 
Take a clean image $x$ and increasingly adding more noise to get an image with pure noise, to understand better see image below.

### Mechanism
We start from the initial image $x$ and, sampling from a Gaussian distribution conditioned by the original data $q(z_t|\bf{x})=\mathcal{N}(\alpha_t\bf{x}, \sigma^2_t\bf{I})$, we generate a noisier image $z_t$ where $t$ is the timestamp that goes from 0 (clean data) to 1 (pure noise) so $t \in [0, 1]$.
As a measure on how our image is noisy, we use the $SNR(t)$, which stands for Signal to Noise Ratio, basically at every timestamp the $SNR(t)$ decreases: $SNR(t) < SNR(s)$ where $t>s$, less is the value noisier is the image (monotonically decreasing), infact it is defined as $SNR(t) = \frac{\alpha_t^2}{\sigma_t^2}$ where $\alpha_t$ is the original signal strenght and $\sigma^2_t$ is the variance of the noise added.

### Code
The formula to sample $z_t$ from the above distribution is $z_t = \alpha(t)x + \sigma(t)\epsilon$ from equation (13).

The implementation in our code is:
```python
def sample_z(self, x, t, noise=None):
        gamma_t = self.gamma(t)
        gamma_t_padded = gamma_t.view(-1, *([1] * (x.ndim - 1)))
        
        alpha_t = get_alpha(gamma_t_padded)
        sigma_t = get_sigma(gamma_t_padded)
        
        if noise is None:
            noise = torch.randn_like(x)
        
        z_t = alpha_t * x + sigma_t * noise
        return z_t, gamma_t, noise
```

$t$ is $\in [0,1]$ since we pass it as 
```python 
i = torch.randint(1, self T + 1, (B,), device=self.device) 
t = i.float() / self.T
```

# Noise Schedule
It determines how the data is corrupted during the forward process. 
In previous works the schedule was fixed, however in our case is learned. To learn the schedule we use a monotonic neural network $\gamma_\eta(t)$, we can link the $SNR(t)$ to the learned schedule: 
$
\sigma_t^2 = \text{sigmoid}(\gamma_\eta(t)) \\
\alpha_t = \sqrt{1-\sigma_t^2} \\
\alpha_t^2 = \text{sigmoid}(-\gamma_\eta(t)) \\
SNR(t) = \exp (-\gamma_\eta(t)) \\
$ 
This way the model can optimize how noise is applied, which helps to minimize the variance of the VLB (Variational Lower Bound) estimator.  

### Code 
```python
def get_alpha(gamma):
    return sqrt(sigmoid(-gamma))

def get_sigma(gamma):
    return sqrt(sigmoid(gamma))

def get_snr(gamma):
    return exp(-gamma)
```

# Reverse Diffusion Process
### Goal 
Sample clean data $x$ starting from pure noise $z_1$.

### Mechanism
We can do that by inverting the forward diffusion process, starting from $t=1$ to $t=T$, basically we train the model such that it learns the probability distribution $p(z_s|z_t)$.

The model starts by defining the noise variable $z_1$ (pure noise image) as a spherical Gaussian $p(z_1) = \mathcal{N}(z_1;0,\bf{I})$:
```python
z = torch.randn((batch_size, *image_shape), device=device)
```

Then in every step we model the probability of a cleaner state $z_s$ given a noisier state $z_t$:
```python
t = steps[i].expand(batch_size)      
s = steps[i + 1].expand(batch_size)  
gamma_t = self.gamma(t)              
gamma_s = self.gamma(s)
```
To do that we use the exact distribution used for the forward diffusion process but using $\hat{x_\theta}(z_t;t)$ instead of $\bf{x}$ which is the output of the denoising model.
To run the reverse diffusion process the models needs to figure out the clean image $x$ from a noisy version $z_t$, we do that with the denoising model $\hat{x_\theta}(z_t;t) = \frac{(z_t-\sigma_t\hat{\epsilon_\theta}(z_t;t))}{\alpha_t}$, however instead of training the network to directly predict the clean image we train a network called noise prediction model $\hat{\epsilon_\theta}(z_t;t)$: 
```python
eps_hat = self.model(z, gamma_t)
```
That basically predicts the noise that was being added to the image such that we can easily recover it.


### Code

In our code we used the semplification of  $p(z_s|z_t) = q(z_s|z_t,x=\hat{x_\theta}(z_t;t))$, formula (34):
$
\\
\mu_\theta(z_t;s,t) = \frac{\alpha_s}{\alpha_t}(z_t+\sigma_t\text{expm1}(\gamma_{\eta}(s)-\gamma_{\eta}(t))\hat{\epsilon}_\theta(z_t;t)) 
\\ 
c = -\text{expm1}(\gamma_\eta(s)-\gamma_\eta(t))
$
``` python                                
c = -expm1(gamma_s_padded - gamma_t_padded)
mean = (alpha_s / alpha_t) * (z - sigma_t * c * eps_hat)
```
$$
\\
z_s = \mu_\theta(z_t;s,t) + \sqrt{\sigma^2_sc}\epsilon
$$
where $\epsilon = \mathcal{N}(0,\bf{I})$
``` python                                
noise = torch.randn_like(z)
z = mean + sigma_s * torch.sqrt(c) * noise
```

# Loss Function

### Goal 
Quantify how well the model reconstructs the data after denoising, balancing signal preservation, noise removal, and reconstruction of discrete pixel values.

### Mechanism

The total loss is composed of three main terms, equation (11):
$$
-VLB(x) = \text{prior loss} + \text{reconstruction loss} + \text{diffusion loss}
$$

- Diffusion loss $L_T(x)$:
    measures how accurately the model predicts the added noise $\hat{\epsilon}_\theta(z_t;t)$ at each time step equation (17):
    $
    \\
    \mathcal{L}(x) = \frac{1}{2}\bf{E}[\gamma_\eta'(t)||\epsilon-\hat{\epsilon}_\theta(z_t;t)||^2_2]
    $

    where $\gamma_\eta'(t) = \frac{d\gamma_\eta(t)}{dt}$

    ```python
    dgamma_dt = (self.gamma_max - self.gamma_min)  
    mse = ((eps - eps_hat) ** 2).mean(dim=(1, 2, 3))
    diffusion_loss = 0.5 * dgamma_dt * mse
    ```
    notice that we used the continuous time model loss and not the discrete time model loss, that's because we pass $t$ as $t = \frac{i}{T} \in [0,1]$:
    ```python
    t = i.float() / self.T
    ```

- Prior loss $D_{KL}[(z_1|x)||p(z_1)]$: 
    Encourages the model to match the distribution of the underlying latent space at the final time step. $q(z_1|x)$ is the distribution fully noised produced by the forward diffusion process, $p(z_1)$ is the target distribution, the KL divergece measure the difference between the 2 distributions where $q(z_1|x) = \mathcal{N}(\mu_q,\sigma^2_q\bf{I})$ and $p(z_1) = \mathcal{N}(0, \bf{I})$. The KL divergence between the 2 distributions is $D_{KL} = 0.5(\text{tr}(\sigma^2_q\bf{I}) + ||\mu_q||^2 - k -\log(\text{det}\sigma^2_q\bf{I}))$, since we know that $\mu_q = \alpha_1 x$ and $\sigma_q^2 = \sigma_1^2$ so $SNR(1) = \alpha_1^2/\sigma^2_1 so \alpha_1^2 = SNR(1)\sigma_1^2$, if we plague everything in the KL divergence we have $D_{KL} (q(z_1|x)|| p(z_1)) = 0.5[\sigma_1^2+(\alpha_1 x)^2 - 1 -\log(\sigma_1^2)]$, we can arrange all the terms as $L_{prior}(x)=\frac{SNR(1)}{SNR(1)+1}\cdot ||x||^2$ (you can find more references here: https://www.themoonlight.io/en/review/demystifying-variational-diffusion-models, https://blog.alexalemi.com/diffusion.html):
    ```python
    gamma_1 = self.gamma(torch.ones(B, device=self.device))
    snr_1 = get_snr(gamma_1)
    prior_factor = (snr_1 / (snr_1 + 1)).view(B, 1, 1, 1)
    prior_loss = (prior_factor * (x_cont ** 2)).mean(dim=(1, 2, 3))
    ```
    
- Reconstruction loss $\bf{E}_{q(z_0|x)}[-\log p(x|z_0)]$: 
    Computes the negative log-likelihood of the original discrete image given the predicted distribution from the final denoised state. This term quantifies how well the model can recover the original data $x$ from a denoised latent $z0$.
    ```python
    t_0 = torch.zeros(B, device=self.device)
    z_0, gamma_0, _ = self.forward_diffusion.sample_z(x_cont, t_0)

    gamma_0_padded = gamma_0.view(B, 1, 1, 1)
    alpha_0 = get_alpha(gamma_0_padded)
    sigma_0 = get_sigma(gamma_0_padded)

    x_vals = 2 * (
        (torch.arange(self.vocab_size, device=self.device).float() + 0.5)
        / self.vocab_size
    ) - 1
    mu_vals = alpha_0.unsqueeze(-1) * x_vals.view(1, 1, 1, 1, -1)

    z_0_exp = z_0.unsqueeze(-1)
    dist_sq = ((z_0_exp - mu_vals) ** 2) / (sigma_0.unsqueeze(-1) ** 2)
    logits = -0.5 * dist_sq

    log_probs = torch.log_softmax(logits, dim=-1)
    x_int_exp = x_int.unsqueeze(-1)
    reconstruction_loss = -log_probs.gather(-1, x_int_exp).squeeze(-1).mean(
        dim=(1, 2, 3)
    )
    ```
