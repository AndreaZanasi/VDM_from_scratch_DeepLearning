import torch
from torch.optim import Adam
import os
from tqdm import tqdm

class Trainer:
    def __init__(self, model, data_provider, config):
        self.model = model
        self.data_provider = data_provider
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = Adam(self.model.parameters(), lr=config.get('lr', 2e-4))
        self.save_dir = config.get('save_dir', './checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            pbar = tqdm(self.data_provider.train, desc=f"Epoch {epoch+1}/{epochs}")
            avg_loss = 0
            
            for i, batch in enumerate(pbar):
                self.optimizer.zero_grad()
                
                if isinstance(batch, (list, tuple)):
                    x, _ = batch
                else:
                    x = batch
                
                # DataProvider yields [-1, 1], VDM expects [0, 1] for discretization
                x = (x + 1) / 2
                
                # VDM handles device movement internally
                loss, metrics = self.model(x)
                
                loss.backward()
                self.optimizer.step()
                
                avg_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                
            print(f"Epoch {epoch+1} Average Loss: {avg_loss / len(self.data_provider.train):.4f}")
            print(f"Metrics: {metrics}")
            self.save_checkpoint(epoch)
            
    def save_checkpoint(self, epoch):
        path = os.path.join(self.save_dir, f"checkpoint_{epoch}.pt")
        torch.save(self.model.state_dict(), path)
