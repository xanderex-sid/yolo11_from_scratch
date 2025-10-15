import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.loss_functions import YOLOLoss

class YOLOTrainer:
    """
    Simplified YOLO trainer for object detection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        cfg: Optional[Dict] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: YOLO model with HeadOD
            train_loader: Training data loader
            val_loader: Validation data loader
            cfg: Configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Default config
        default_cfg = {
            'epochs': 100,
            'lr': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'nc': 80,
            'save_dir': './runs/train',
            'name': 'exp',
            'batch_size': 16,
        }
        
        self.cfg = {**default_cfg, **(cfg or {})}
        self.device = torch.device(self.cfg['device'])
        self.model.to(self.device)
        
        # Initialize loss
        self.criterion = YOLOLoss(nc=self.cfg['nc'])
        
        # Initialize optimizer
        self.optimizer = self._build_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._build_scheduler()
        
        # Metrics tracking
        self.epoch = 0
        self.best_fitness = 0.0
        
        # Create save directory
        self.save_dir = Path(self.cfg['save_dir']) / self.cfg['name']
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer with parameter groups."""
        g = [], [], []  # optimizer parameter groups: [biases, weights (no decay), weights (decay)]
        
        for v in self.model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                g[0].append(v.bias)
            if isinstance(v, nn.BatchNorm2d):
                g[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                g[2].append(v.weight)
        
        optimizer = optim.SGD(
            g[0],
            lr=self.cfg['lr'],
            momentum=self.cfg['momentum'],
            nesterov=True
        )
        
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})
        optimizer.add_param_group({'params': g[2], 'weight_decay': self.cfg['weight_decay']})
        
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        lf = lambda x: (1 - x / self.cfg['epochs']) * (1.0 - 0.01) + 0.01
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
    
    def train_one_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        mloss = torch.zeros(3, device=self.device)  # box, cls, dfl
        
        for i, (imgs, targets) in pbar:
            # Warmup
            ni = i + len(self.train_loader) * self.epoch
            if ni <= self.cfg['warmup_epochs'] * len(self.train_loader):
                xi = [0, self.cfg['warmup_epochs'] * len(self.train_loader)]
                for j, x in enumerate(self.optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [self.cfg['warmup_bias_lr'] if j == 0 else 0.0, 
                                                   x['initial_lr'] * self.scheduler.get_last_lr()[0]])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [self.cfg['warmup_momentum'], 
                                                            self.cfg['momentum']])
            
            # Forward
            imgs = imgs.to(self.device).float() / 255.0
            targets = targets.to(self.device)
            
            preds = self.model(imgs)
            loss, loss_items = self.criterion(preds, targets, self.model)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            mloss = (mloss * i + loss_items) / (i + 1)
            
            # Print
            pbar.set_description(
                f"Epoch {self.epoch}/{self.cfg['epochs']} | "
                f"Box: {mloss[0]:.4f} | Cls: {mloss[1]:.4f} | DFL: {mloss[2]:.4f}"
            )
        
        return {
            'train/box_loss': mloss[0].item(),
            'train/cls_loss': mloss[1].item(),
            'train/dfl_loss': mloss[2].item(),
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        pbar = tqdm(self.val_loader, desc='Validating')
        mloss = torch.zeros(3, device=self.device)
        
        with torch.no_grad():
            for i, (imgs, targets) in enumerate(pbar):
                imgs = imgs.to(self.device).float() / 255.0
                targets = targets.to(self.device)
                
                preds = self.model(imgs)
                loss, loss_items = self.criterion(preds, targets, self.model)
                
                mloss = (mloss * i + loss_items) / (i + 1)
        
        return {
            'val/box_loss': mloss[0].item(),
            'val/cls_loss': mloss[1].item(),
            'val/dfl_loss': mloss[2].item(),
        }
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.cfg['epochs']} epochs...")
        print(f"Device: {self.device}")
        print(f"Save directory: {self.save_dir}")
        
        for epoch in range(self.cfg['epochs']):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_one_epoch()
            
            # Update scheduler
            self.scheduler.step()
            
            # Validate
            val_metrics = self.validate()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Calculate fitness (simplified)
            fitness = -metrics.get('val/box_loss', metrics['train/box_loss'])
            
            # Save checkpoint
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.save_checkpoint('best.pt')
            
            # Save last checkpoint
            if (epoch + 1) % 10 == 0 or epoch == self.cfg['epochs'] - 1:
                self.save_checkpoint('last.pt')
            
            # Print metrics
            print(f"\nEpoch {epoch} metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            print(f"  Fitness: {fitness:.4f}\n")
        
        print(f"Training complete. Best fitness: {self.best_fitness:.4f}")
        print(f"Results saved to {self.save_dir}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        
        torch.save(ckpt, self.save_dir / filename)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.epoch = ckpt['epoch']
        self.best_fitness = ckpt.get('best_fitness', 0.0)
        
        print(f"Loaded checkpoint from {path}")


# Example usage
if __name__ == "__main__":
    """
    Example of how to use the trainer.
    You need to provide:
    1. Your model with HeadOD
    2. Data loaders with format: (images, targets)
       - images: [batch, 3, H, W]
       - targets: [num_targets, 6] where each row is [batch_idx, class, x, y, w, h]
    """
    
    # Dummy example (replace with your actual model and data)
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            # Add your HeadOD here
            
        def forward(self, x):
            x = self.conv(x)
            # Return list of predictions from different scales
            return [x]
    
    # Create model
    model = DummyModel()
    
    # Create dummy data loaders (replace with your actual data)
    # train_loader = DataLoader(your_train_dataset, batch_size=16, shuffle=True)
    # val_loader = DataLoader(your_val_dataset, batch_size=16)
    
    # Configuration
    cfg = {
        'epochs': 100,
        'lr': 0.01,
        'batch_size': 16,
        'nc': 80,  # number of classes
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    # Create trainer
    # trainer = YOLOTrainer(model, train_loader, val_loader, cfg)
    
    # Start training
    # trainer.train()