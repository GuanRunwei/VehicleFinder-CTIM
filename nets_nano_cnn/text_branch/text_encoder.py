import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from nets_nano_cnn.text_branch.ds_conv_text import ds_basic_conv


class text_encoder(nn.Module):
    def __init__(self, in_channels):
        super(text_encoder, self).__init__()
        self.in_channels = in_channels

        # ====== small kernel branch ==========================
        self.small_k_branch = nn.Sequential(
            ds_basic_conv(in_channels=1, out_channels=16, kernel_size=(1, 5), padding=(0, 0), stride=(1, 3)),
            ds_basic_conv(in_channels=16, out_channels=64, kernel_size=(1, 4), padding=(0, 0), stride=(1, 2)),
            ds_basic_conv(in_channels=64, out_channels=128, kernel_size=(1, 3), padding=(0, 1), stride=(1, 1)),
            ds_basic_conv(in_channels=128, out_channels=256, kernel_size=(1, 3), padding=(0, 1), stride=(1, 1)),
            ds_basic_conv(in_channels=256, out_channels=512, kernel_size=(1, 3), padding=(0, 1), stride=(1, 1))
        )
        self.middle_k_branch = nn.Sequential(
            ds_basic_conv(in_channels=1, out_channels=16, kernel_size=(2, 25), stride=1, padding=0),
            ds_basic_conv(in_channels=16, out_channels=64, kernel_size=(1, 52), stride=1, padding=0),
            ds_basic_conv(in_channels=64, out_channels=128, kernel_size=(1, 52), stride=1, padding=0),
            ds_basic_conv(in_channels=128, out_channels=256, kernel_size=(1, 52), stride=1, padding=0),
            ds_basic_conv(in_channels=256, out_channels=512, kernel_size=(1, 52), stride=1, padding=0),
        )
        self.large_k_branch = nn.Sequential(
            ds_basic_conv(in_channels=1, out_channels=16, kernel_size=(3, 2), stride=(1, 1), padding=0),
            ds_basic_conv(in_channels=16, out_channels=64, kernel_size=(1, 2), stride=(1, 2), padding=0),
            ds_basic_conv(in_channels=64, out_channels=128, kernel_size=(1, 2), stride=(1, 1), padding=0),
            ds_basic_conv(in_channels=128, out_channels=256, kernel_size=(1, 3), stride=(1, 1), padding=0),
            ds_basic_conv(in_channels=256, out_channels=512, kernel_size=(1, 3), stride=(1, 1), padding=0),
        )
        self.huge_k_res = nn.Sequential(
            ds_basic_conv(in_channels=1, out_channels=256, kernel_size=(3, 13), stride=(1, 2), padding=0),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0),
        )

    def forward(self, x):
        x_middle, x_large, x_res = x, x, x

        x_middle = self.middle_k_branch(x_middle).flatten(2)
        x_large = self.large_k_branch(x_large).flatten(2)
        x_res = self.huge_k_res(x_res).flatten(2)
        x_output = x_middle + x_large + x_res
        return x_output



if __name__ == '__main__':
    te = text_encoder(in_channels=1)
    input_map = torch.randn(16, 1, 3, 300)
    output_map = te(input_map)
    print(output_map.shape)