from src.modules import Conv, C3k2
import torch.nn as nn

class Backbone(nn.Module):
    def __init__(self, d, w, mc):
        super().__init__()
        self.cv_0 = Conv(3, (min(64, mc) * w), k=3, s=2)
        self.cv_1 = Conv((min(64, mc) * w), (min(128, mc) * w), k=3, s=2)
        self.c3k2_2 = C3k2((min(128, mc) * w), (min(256, mc) * w), n=2*d, c3k=False, e=0.25)
        self.cv_3 = Conv((min(256, mc) * w), (min(256, mc) * w), k=3, s=2)
        self.c3k2_4 = C3k2((min(256, mc) * w), (min(512, mc) * w), n=2*d, c3k=False, e=0.25)
        self.cv_5 = Conv((min(512, mc) * w), (min(512, mc) * w), k=3, s=2)
        self.c3k2_6 = C3k2((min(512, mc) * w), (min(512, mc) * w), n=2*d, c3k=True)
        self.cv_7 = Conv((min(512, mc) * w), (min(1024, mc) * w), k=3, s=2)
        self.c3k2_8 = C3k2((min(1024, mc) * w), (min(1024, mc) * w), n=2*d, c3k=True)

    def forward(self, x):
        x = self.cv_0(x)
        x = self.cv_1(x)
        x = self.c3k2_2(x)
        x = self.cv_3(x)
        f1 = self.c3k2_4(x)
        x = self.cv_5(f1)
        f2 = self.c3k2_6(x)
        x = self.cv_7(f2)
        f3 = self.c3k2_8(x)
        return f1, f2, f3