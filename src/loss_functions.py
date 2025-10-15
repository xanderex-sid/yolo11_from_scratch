import torch
import torch.nn as nn
from typing import List, Tuple


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """
        Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

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