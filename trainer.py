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
        print(f"VDM Training Started")
        print(f"{'='*70}\n")
        
        for epoch in range(epochs):
            # 1. Train
            train_loss = self.train_epoch(epoch + 1, epochs)
            self.train_losses.append(train_loss)
            
            # 2. Validate (New)
            if (epoch + 1) % self.config.get('validate_every', 1) == 0:
                val_loss = self.validate(epoch + 1)
            
            # 3. Scheduler Step
            self.scheduler.step()
            
            # 4. Save Checkpoints
            self.handle_checkpoints(train_loss, epoch + 1)
            
            # 5. Visualizations (Samples & Schedule)
            if self.config.get('use_wandb', False):
                self.log_visualizations(epoch + 1)

            # Print Summary
            print(f"Epoch {epoch+1}: Train {train_loss:.4f} | Best {self.best_loss:.4f}")

    def train_epoch(self, epoch, total_epochs):
        self.model.train()
        pbar = tqdm(self.data_provider.train, desc=f"Train Ep {epoch}/{total_epochs}", ncols=100)
        
        total_loss = 0
        steps = 0
        
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
            
            # Metrics
            current_loss = loss.item()
            total_loss += current_loss
            steps += 1
            
            if self.ema_loss is None:
                self.ema_loss = current_loss
            else:
                self.ema_loss = self.ema_decay * self.ema_loss + (1 - self.ema_decay) * current_loss
            
            # --- WANDB LOGGING (Per Step) ---
            if self.config.get('use_wandb', False) and steps % self.config.get('wandb_log_freq', 50) == 0:
                wandb.log({
                    "train/total_loss": current_loss,
                    "train/diffusion_loss": metrics.get("diffusion", 0),
                    "train/prior_loss": metrics.get("prior", 0),
                    "train/recon_loss": metrics.get("reconstruction", 0),
                    "train/lr": self.optimizer.param_groups[0]['lr'],
                    "epoch": epoch
                })

            pbar.set_postfix({'loss': f'{current_loss:.3f}', 'ema': f'{self.ema_loss:.3f}'})
        
        return total_loss / steps if steps > 0 else 0

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