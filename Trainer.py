import torch
from torch.optim import AdamW
import os
from tqdm import tqdm
import time

class Trainer:
    def __init__(self, model, data_provider, config):
        self.model = model
        self.data_provider = data_provider
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        # Use AdamW with paper's settings
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
        
        # For tracking progress
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
            
            # Training
            train_loss, train_metrics = self.train_epoch(epoch + 1, epochs)
            self.train_losses.append(train_loss)
            
            # Step scheduler
            self.scheduler.step()
            
            # Compute statistics
            epoch_time = time.time() - epoch_start
            elapsed_time = time.time() - start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = epochs - (epoch + 1)
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_hours = int(eta_seconds // 3600)
            eta_mins = int((eta_seconds % 3600) // 60)
            
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
            self.handle_checkpoints(train_loss, epoch + 1)
            
            # Periodic detailed status
            if (epoch + 1) % 10 == 0:
                self.print_progress_summary(epoch + 1, epochs)
    
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
            
            # Handle batch format
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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            current_loss = loss.item()
            total_loss += current_loss
            steps += 1
            last_metrics = metrics
            
            # Update EMA
            if self.ema_loss is None:
                self.ema_loss = current_loss
            else:
                self.ema_loss = self.ema_decay * self.ema_loss + (1 - self.ema_decay) * current_loss
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{current_loss:.3f}',
                'ema': f'{self.ema_loss:.3f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.1e}'
            })
        
        avg_loss = total_loss / steps if steps > 0 else 0
        return avg_loss, last_metrics
    
    def handle_checkpoints(self, current_loss, epoch):
        # Always save last model
        last_path = os.path.join(self.save_dir, "last_model.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': current_loss,
            'ema_loss': self.ema_loss,
            'train_losses': self.train_losses,
        }, last_path)
        
        # Save best model
        if current_loss < self.best_loss:
            improvement = self.best_loss - current_loss
            self.best_loss = current_loss
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': current_loss,
                'ema_loss': self.ema_loss,
                'train_losses': self.train_losses,
            }, best_path)
            print(f"  ✓ New best model! (improved by {improvement:.4f} BPD)")
    
    def print_progress_summary(self, current_epoch, total_epochs):
        """Print a detailed summary every N epochs"""
        if len(self.train_losses) < 10:
            return
        
        recent_losses = self.train_losses[-10:]
        trend = "↓ improving" if recent_losses[-1] < recent_losses[0] else "→ plateau"
        
        print(f"\n{'='*70}")
        print(f"Progress Summary (Epoch {current_epoch}/{total_epochs}):")
        print(f"  Current Loss:  {self.train_losses[-1]:.4f} BPD")
        print(f"  Best Loss:     {self.best_loss:.4f} BPD")
        print(f"  Initial Loss:  {self.train_losses[0]:.4f} BPD")
        print(f"  Improvement:   {self.train_losses[0] - self.train_losses[-1]:.4f} BPD")
        print(f"  Recent Trend:  {trend}")
        print(f"  Target (Paper): 2.67 BPD")
        print(f"  Gap to Target: {self.train_losses[-1] - 2.67:+.4f} BPD")
        print(f"{'='*70}\n")
    
    def load_checkpoint(self, path):
        """Load a checkpoint to resume training"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.ema_loss = checkpoint.get('ema_loss', None)
        self.best_loss = checkpoint['loss']
        print(f"Checkpoint loaded from {path}")
        print(f"Resuming from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f} BPD")
        return checkpoint['epoch']