import torch
from torch.optim import AdamW
import os
from tqdm import tqdm
import time
import wandb
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

class Trainer:
    def __init__(self, model, data_provider, config):
        self.model = model
        self.data_provider = data_provider
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.use_wandb = config.get('use_wandb', False)
        
        # Use AdamW
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=config.get('lr', 2e-4),
            betas=(0.9, 0.99),
            weight_decay=0.01
        )
        
        # Cosine learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 200),
            eta_min=config.get('lr', 2e-4) * 0.1
        )
        
        self.save_dir = config.get('save_dir', './checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.best_loss = float('inf')
        self.train_losses = []
        self.ema_loss = None
        self.ema_decay = 0.99

    def train(self, epochs):
        print(f"\n{'='*70}")
        print(f"Training Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {self.config.get('batch_size', 64)}")
        print(f"  Learning Rate: {self.config.get('lr', 2e-4):.2e}")
        print(f"  Device: {self.device}")
        print(f"  Target: ~2.67 BPD (paper baseline for T=1000)")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            
            epoch_start = time.time()
            # 1. Train
            train_loss, train_metrics = self.train_epoch(epoch + 1, epochs)
            self.train_losses.append(train_loss)
            
            # 2. Validate (New)
            if (epoch + 1) % self.config.get('validate_every', 1) == 0:
                val_loss = self.validate(epoch + 1)
            
            # 3. Scheduler Step
            self.scheduler.step()
            
            # Compute statistics
            epoch_time = time.time() - epoch_start
            elapsed_time = time.time() - start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = epochs - (epoch + 1)
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_hours = int(eta_seconds // 3600)
            eta_mins = int((eta_seconds % 3600) // 60)
            
            # Log to wandb
            
            if self.use_wandb:
                wandb.log({
                    "train/total_loss": train_loss,
                    "train/reconstruction_loss": train_metrics.get("reconstruction", 0),
                    "train/epoch/diffusion_loss": train_metrics.get("diffusion", 0),
                    "train/epoch/prios_loss": train_metrics.get("prior", 0),
                    "train/epoch/ema_loss": self.ema_loss,
                    "train/epoch/lr": self.optimizer.param_groups[0]["lr"],
                    "train/epoch/time_sec": epoch_time
                })
                            
            # Print epoch summary
            print(f"\n{'─'*70}")
            print(f"Epoch {epoch+1}/{epochs} Complete:")
            print(f"  Train Loss:    {train_loss:.4f} BPD")
            print(f"  EMA Loss:      {self.ema_loss:.4f} BPD")
            print(f"  Best Loss:     {self.best_loss:.4f} BPD")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Time: {epoch_time:.1f}s | ETA: {eta_hours}h {eta_mins}m")
            print(f"{'─'*70}\n")

            
            # Save checkpoints
            # 4. Save Checkpoints
            self.handle_checkpoints(train_loss, epoch + 1)
            
            # 5. Visualizations (Samples & Schedule)
            if self.config.get('use_wandb', False):
                self.log_visualizations(epoch + 1)

            # # Periodic detailed status
            # if (epoch + 1) % 10 == 0:
            #     self.print_progress_summary(epoch + 1, epochs)
            
            if (epoch + 1) % self.config.get('sample_every_epochs', 5) == 0:
                self.log_reconstructed_samples(epoch + 1)

    def train_epoch(self, epoch, total_epochs):
        self.model.train()
        
        pbar = tqdm(
            self.data_provider.train, 
            desc=f"Epoch {epoch}/{total_epochs}",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        total_loss = 0
        steps = 0
        last_metrics = {}
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            if isinstance(batch, (list, tuple)):
                x, _ = batch
            else:
                x = batch
            
            x = x.to(self.device)
            x = (x + 1) / 2  # Normalize to [0, 1]
            
            # Forward pass
            loss, metrics = self.model(x)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            current_loss = loss.item()
            total_loss += current_loss
            steps += 1
            last_metrics = metrics
            
            if self.ema_loss is None:
                self.ema_loss = current_loss
            else:
                self.ema_loss = self.ema_decay * self.ema_loss + (1 - self.ema_decay) * current_loss
            
            # --- WANDB LOGGING (Per Step) ---
            # if self.config.get('use_wandb', False) and steps % self.config.get('wandb_log_freq', 50) == 0:
            #     wandb.log({
            #         "train/total_loss": current_loss,
            #         "train/diffusion_loss": metrics.get("diffusion", 0),
            #         "train/prior_loss": metrics.get("prior", 0),
            #         "train/recon_loss": metrics.get("reconstruction", 0),
            #         "train/lr": self.optimizer.param_groups[0]['lr'],
            #         "epoch": epoch
            #     })

            #     # Log any model-provided metrics
            #     for k, v in metrics.items():
            #         if isinstance(v, (int, float)):
            #             log_data[f"train/step/{k}"] = v

            #     wandb.log(log_data, step=steps)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{current_loss:.3f}',
                'ema': f'{self.ema_loss:.3f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.1e}'
            })
        
        avg_loss = total_loss / steps if steps > 0 else 0
        return avg_loss, last_metrics

    def validate(self, epoch):
        """Run validation on test set"""
        self.model.eval()
        total_val_loss = 0
        steps = 0
        
        # Only run a subset to save time if dataset is huge
        limit_batches = 50 
        
        with torch.no_grad():
            for i, batch in enumerate(self.data_provider.test):
                if i >= limit_batches: break
                
                if isinstance(batch, (list, tuple)):
                    x, _ = batch
                else:
                    x = batch
                x = x.to(self.device)
                x = (x + 1) / 2
                
                loss, metrics = self.model(x)
                total_val_loss += loss.item()
                steps += 1
        
        avg_val_loss = total_val_loss / steps if steps > 0 else 0
        
        if self.config.get('use_wandb', False):
            wandb.log({
                "val/total_loss": avg_val_loss,
                "val/diffusion_loss": metrics.get("diffusion", 0),
                "val/prior_loss": metrics.get("prior", 0),
                "val/reconstruction_loss": metrics.get("reconstruction", 0),
                "epoch": epoch
            })
            
        return avg_val_loss

    def log_visualizations(self, epoch):
        """Handle Sampling and Schedule Plotting"""
        
        # 1. Periodic Image Sampling
        if epoch % self.config.get('sample_every_epochs', 5) == 0:
            self.model.eval()
            with torch.no_grad():
                # Generate Samples
                samples = self.model.reverse_diffusion.sample(
                    batch_size=16, # Small batch for visualization
                    shape=(3, 32, 32),
                    T=self.config.get('T', 1000),
                    device=self.device
                )
                
                # Create grid
                grid = make_grid(samples, nrow=4, normalize=False)
                
                # Log to WandB
                wandb.log({
                    "generated_samples": wandb.Image(grid, caption=f"Epoch {epoch}"),
                    "epoch": epoch
                })
        
        # 2. Variance Schedule Visualization
        if epoch % self.config.get('plot_schedule_every', 10) == 0:
            self.plot_variance_schedule(epoch)

    def plot_variance_schedule(self, epoch):
        """Plots gamma(t) and SNR(t) to visualize the learned schedule"""
        t = torch.linspace(0, 1, 1000, device=self.device)
        with torch.no_grad():
            gamma = self.model.gamma(t)
            snr = torch.exp(-gamma)
        
        t_cpu = t.cpu().numpy()
        gamma_cpu = gamma.cpu().numpy()
        snr_cpu = snr.cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Plot Gamma
        ax1.plot(t_cpu, gamma_cpu)
        ax1.set_title("Learned Gamma(t)")
        ax1.set_xlabel("t")
        ax1.set_ylabel("gamma")
        ax1.grid(True)
        
        # Plot SNR
        ax2.plot(t_cpu, snr_cpu)
        ax2.set_title("Signal-to-Noise Ratio (SNR)")
        ax2.set_xlabel("t")
        ax2.set_ylabel("SNR")
        ax2.set_yscale('log')
        ax2.grid(True)
        
        wandb.log({
            "variance_schedule": wandb.Image(fig),
            "epoch": epoch
        })
        plt.close(fig)

    def handle_checkpoints(self, current_loss, epoch):
        # Save last
        last_path = os.path.join(self.save_dir, self.config.get('last_model_path', 'last_model.pt'))
        self._save(last_path, current_loss, epoch)
        
        # Save best
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            best_path = os.path.join(self.save_dir, self.config.get('best_model_path', 'best_model.pt'))
            self._save(best_path, current_loss, epoch)
            
            # Log best metric to wandb
            if self.config.get('use_wandb', False):
                wandb.run.summary["best_loss"] = self.best_loss
                # Create or overwrite artifact
                artifact = wandb.Artifact(
                    name="vdm-best-model",
                    type="model",
                    description="Best VDM checkpoint based on training loss",
                    metadata={"epoch": epoch, "loss": self.best_loss}
                )
                if os.path.exists(best_path):
                    artifact.add_file(best_path)
                
                wandb.log_artifact(artifact, aliases=["latest"])

    def _save(self, path, loss, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'ema_loss': self.ema_loss,
            'train_losses': self.train_losses,
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.ema_loss = checkpoint.get('ema_loss', None)
        self.best_loss = checkpoint['loss']
        print(f"Resuming from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
        return checkpoint['epoch']
    
    def log_reconstructed_samples(self, epoch, num_samples=8, t_fraction=0.7):
        """
        Log original, noised, and reconstructed samples from the validation set to WandB.

        Args:
            epoch (int): current epoch
            num_samples (int): number of images to reconstruct
            t_fraction (float): fraction of total diffusion steps to forward-noise
        """
        self.model.eval()
        val_iter = iter(self.data_provider.test)
        batch = next(val_iter)  # Take one batch
        if isinstance(batch, (list, tuple)):
            x, _ = batch
        else:
            x = batch

        x = x[:num_samples].to(self.device)
        batch_size = x.shape[0]

        # Normalize to [0, 1] for plotting
        x_vis = (x + 1) / 2  

        with torch.no_grad():
            # 1. Add forward noise at timestep t_fraction
            t = torch.tensor(t_fraction, device=self.device)  # scalar tensor
            z_t, _, eps = self.model.forward_diffusion(x, t)

            # 2. Reverse diffusion starting from noisy z_t
            recon = self.model.reverse_diffusion.sample_from_noisy(
                z_t=z_t,
                t_start=t,
                T=int(self.model.T * t_fraction),
                device=self.device
            )

            # Normalize noised and reconstructed images
            z_vis = torch.clamp((z_t + 1) / 2, 0, 1)
            recon_vis = torch.clamp((recon + 1) / 2, 0, 1)

            # 3. Stack images: original | noised | reconstructed
            # Each image triplet will occupy one row
            grid_rows = []
            for i in range(batch_size):
                row = torch.stack([x_vis[i], z_vis[i], recon_vis[i]], dim=0)
                row_grid = make_grid(row, nrow=3, padding=2, normalize=False)
                grid_rows.append(row_grid)

            grid = torch.cat(grid_rows, dim=1)  # Stack rows vertically

            # 4. Log to WandB
            wandb.log({
                "reconstructed_samples": wandb.Image(grid, caption=f"Epoch {epoch}"),
                "epoch": epoch
            })
