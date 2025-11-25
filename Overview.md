# Forward Diffusion Process

### Goal 
Take a clean image $x$ and increasingly adding more noise to get an image with pure noise, to understand better see image below.

### Mechanism
We start from the initial image $x$ and, sampling from a Gaussian distribution conditioned by the original data $q(z_t|\bf{x})=\mathcal{N}(\alpha_t\bf{x}, \sigma^2_t\bf{I})$, we generate a noisier image $z_t$ where $t$ is the timestamp that goes from 0 (clean data) to 1 (pure noise) so $t \in [0, 1]$.
As a measure on how our image is noisy, we use the $SNR(t)$, which stands for Signal to Noise Ratio, basically at every timestamp the $SNR(t)$ decreases: $SNR(t) < SNR(s)$ where $t>s$, less is the value noisier is the image (monotonically decreasing), infact it is defined as $SNR(t) = \frac{\alpha_t^2}{\sigma_t^2}$ where $\alpha_t$ is the original signal strenght and $\sigma^2_t$ is the variance of the noise added.

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

# Reverse Diffusion Process
### Goal 
Sample clean data $x$ starting from pure noise $z_1$.

### Mechanism
We can do that by inverting the forward diffusion process, starting from $t=1$ to $t=0$, basically we train the model such that it learns the probability distribution $p(z_s|z_t)$.

The model starts by defining the noise variable $z_1$ (pure noise image) as a spherical Gaussian $p(z_1) = \mathcal{N}(z_1;0,\bf{I})$, then in every step we model the probability of a cleaner state $z_s$ given a noisier state $z_t$, to do that we use the exact distribution used for the forward diffusion process but using $\hat{x_\theta}(z_t;t)$ instead of $\bf{x}$ which is the output of the denoising model: $p(z_s|z_t) = q(z_s|z_t,x=\hat{x_\theta}(z_t;t))$.

To run the reverse diffusion process the models needs to figure out the clean image $x$ from a noisy version $z_t$, we do that with the denoising model $\hat{x_\theta}(z_t;t) = \frac{(z_t-\sigma_t\hat{\epsilon_\theta}(z_t;t))}{\alpha_t}$, however instead of training the network to directly predict the clean image we train a network called noise prediction model $\hat{\epsilon_\theta}(z_t;t)$, that basically predicts the noise that was being added to the image such that we can easily recover it.

Our code for the sampling of the next cleaner image $z_s$ given the previous noisy image $z_t$
```python
def ancestral_sampling(self, z, t, s):
    gamma_t = self.gamma(t)
    gamma_s = self.gamma(s)
    c = -expm1(gamma_s - gamma_t)
    alpha_t = self.get_alpha(gamma_t)
    alpha_s = self.get_alpha(gamma_s)
    sigma_t = self.get_sigma(gamma_t)
    predicted_noise = self.model(z, gamma_t)

    mean = alpha_s/alpha_t * (z - sigma_t * c * predicted_noise)
    scale = sqrt((1-alpha_s) * c)
    
    return mean + scale * torch.rand_like(z)

```

We used the formula for the ancestral sampling in appendix A.4 formula 34.