import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class depth_seperate_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 3), stride=1, padding=1):
        super(depth_seperate_conv, self).__init__()
        self.deep_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, x):
        output = self.deep_conv(x)
        output = self.point_conv(output)
        return output


class ds_basic_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=2, padding=1):
        super(ds_basic_conv, self).__init__()
        self.ds_conv = depth_seperate_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x):
        output = self.ds_conv(x)
        layernorm = nn.LayerNorm(output.shape[-1]).to(device)
        output = layernorm(output).to(device)
        output = self.batchnorm(output)
        output = self.activation(output)
        return output


if __name__ == '__main__':
    input_map = torch.randn(16, 1, 3, 300)
    bc = nn.Sequential(
        ds_basic_conv(in_channels=1, out_channels=256, kernel_size=(3, 13), stride=(1, 2), padding=0),
        ds_basic_conv(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=0),

        # ds_basic_conv(in_channels=32, out_channels=64, input_size=75),
        # ds_basic_conv(in_channels=64, out_channels=128, input_size=38),
        # ds_basic_conv(in_channels=128, out_channels=256, input_size=19)
    )
    output_map = bc(input_map)
    # output_map = bc3(output_map)
    print(output_map.shape)

