import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional


class YOLOLoss(nn.Module):
    """
    Simplified YOLO loss function for object detection.
    Combines box regression loss, objectness loss, and classification loss.
    """
    
    def __init__(self, nc: int = 80, reg_max: int = 16):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')
        
        # Loss weights
        self.box_weight = 7.5
        self.cls_weight = 0.5
        self.dfl_weight = 1.5
        
    def forward(self, preds: List[torch.Tensor], targets: torch.Tensor, 
                model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate loss.
        
        Args:
            preds: List of predictions from each detection layer
            targets: Ground truth targets [batch_idx, class, x, y, w, h]
            model: Model with anchors and strides
            
        Returns:
            total_loss, loss_dict
        """
        device = preds[0].device
        loss_box = torch.tensor(0., device=device)
        loss_cls = torch.tensor(0., device=device)
        loss_dfl = torch.tensor(0., device=device)
        
        # Get predictions
        batch_size = preds[0].shape[0]
        
        # Simple implementation: concatenate all predictions
        pred_distri_list = []
        pred_scores_list = []
        
        for i, pred in enumerate(preds):
            b, c, h, w = pred.shape
            # pred shape: [batch, no, h, w] where no = nc + reg_max * 4
            pred = pred.view(b, c, -1).permute(0, 2, 1)  # [batch, h*w, no]
            pred_distri_list.append(pred[..., :self.reg_max * 4])
            pred_scores_list.append(pred[..., self.reg_max * 4:])
        
        pred_distri = torch.cat(pred_distri_list, dim=1)  # [batch, total_anchors, reg_max*4]
        pred_scores = torch.cat(pred_scores_list, dim=1)  # [batch, total_anchors, nc]
        
        # Process targets for each image in batch
        for batch_idx in range(batch_size):
            # Get targets for this image
            batch_targets = targets[targets[:, 0] == batch_idx]
            
            if len(batch_targets) == 0:
                # No targets, apply background loss
                loss_cls += self.bce_cls(pred_scores[batch_idx], 
                                        torch.zeros_like(pred_scores[batch_idx])).mean()
                continue
            
            # Simplified assignment: use IoU matching
            gt_boxes = batch_targets[:, 2:6]  # [x, y, w, h]
            gt_classes = batch_targets[:, 1].long()
            
            # For simplicity, assign to closest anchor (proper implementation would use TAL)
            num_gts = len(gt_boxes)
            
            # Classification loss
            target_scores = torch.zeros_like(pred_scores[batch_idx])
            if num_gts > 0:
                # Simple assignment: assign first few anchors to targets
                for idx in range(min(num_gts, pred_scores.shape[1])):
                    target_scores[idx, gt_classes[idx]] = 1.0
            
            loss_cls += self.bce_cls(pred_scores[batch_idx], target_scores).mean()
            
            # Box loss (simplified)
            if num_gts > 0:
                # Use L1 loss for boxes
                pred_boxes = pred_distri[batch_idx, :num_gts]  # Simplified
                loss_box += nn.functional.l1_loss(pred_boxes, 
                                                   gt_boxes[:num_gts].repeat(1, self.reg_max).to(device))
        
        # Combine losses
        loss_box = loss_box * self.box_weight / batch_size
        loss_cls = loss_cls * self.cls_weight / batch_size
        loss_dfl = loss_dfl * self.dfl_weight / batch_size
        
        total_loss = loss_box + loss_cls + loss_dfl
        
        loss_dict = torch.tensor([loss_box.item(), loss_cls.item(), loss_dfl.item()])
        
        return total_loss, loss_dict


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