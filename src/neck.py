from src.modules import SPPF, C2PSA, C3k2, Conv
import torch
import torch.nn as nn

class Neck(nn.Module):
    def __init__(self, d, w, mc):
        super().__init__()
        self.sppf_9 = SPPF(int(min(1024, mc) * w), int(min(1024, mc) * w), k=5)
        self.c2psa_10 = C2PSA(int(min(1024, mc) * w), int(min(1024, mc) * w), n=int(2*d))
        self.upsample_11 = nn.ConvTranspose2d(int(min(1024, mc) * w), int(min(1024, mc) * w), kernel_size=2, stride=2)
        # 12th step is concatenation of 6th step and upsampled 11th step, see it in forward function.
        self.c3k2_13 = C3k2(int(min(1024, mc) * w) + int(min(512, mc) * w), int(min(512, mc) * w), n=int(2*d), c3k=False)
        self.upsample_14 = nn.ConvTranspose2d(int(min(512, mc) * w), int(min(512, mc) * w), kernel_size=2, stride=2)
        # 15th step is concatenation of 4th step and upsampled 14th step, see it in forward function.
        self.c3k2_16 = C3k2(int(min(512, mc) * w) + int(min(512, mc) * w), int(min(256, mc) * w), n=int(2*d), c3k=False)
        self.conv_17 = Conv(int(min(256, mc) * w), int(min(256, mc) * w), k=3, s=2)
        # 18th step is concatenation of 17th step and 13th step, see it in forward function.
        self.c3k2_19 = C3k2(int(min(256, mc) * w) + int(min(512, mc) * w), int(min(512, mc) * w), n=int(2*d), c3k=False)
        self.conv_20 = Conv(int(min(512, mc) * w), int(min(512, mc) * w), k=3, s=2)
        # 21th step is concatenation of 20th step and 10th step, see it in forward function.
        self.c3k2_22 = C3k2(int(min(512, mc) * w) + int(min(1024, mc) * w), int(min(1024, mc) * w), n=int(2*d), c3k=True)

    def forward(self, backbone_out_4, backbone_out_6, backbone_out_8):
        x = self.sppf_9(backbone_out_8)
        x = self.c2psa_10(x)
        x = self.upsample_11(x)
        x = torch.cat((x, backbone_out_6), dim=1) # 12th step
        x = self.c3k2_13(x)
        x = self.upsample_14(x)
        x = torch.cat((x, backbone_out_4), dim=1) # 15th step
        out_16 = self.c3k2_16(x)
        x = self.conv_17(out_16)
        x = torch.cat((x, backbone_out_6), dim=1) # 18th step
        out_19 = self.c3k2_19(x)
        x = self.conv_20(out_19)
        x = torch.cat((x, backbone_out_8), dim=1) # 21th step
        out_22 = self.c3k2_22(x)
        return out_16, out_19, out_22