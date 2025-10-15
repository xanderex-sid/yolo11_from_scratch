from src.backbone import Backbone
from src.neck import Neck
from src.heads import HeadOD
import torch.nn as nn

class Yolo11ObjectDetectionModel(nn.Module):
    def __init__(self, d=0.5, w=0.25, mc=1024, nc=80):
        super().__init__()
        self.backbone = Backbone(d, w, mc)
        self.neck = Neck(d, w, mc)
        self.head = HeadOD(nc, ch=(int(min(256, mc) * w), int(min(512, mc) * w), int(min(1024, mc) * w)))

    def forward(self, x):
        f1, f2, f3 = self.backbone(x)
        n_out1, n_out2, n_out3 = self.neck(f1, f2, f3)
        out = self.head([n_out1, n_out2, n_out3])
        return out