import torch
import torch.nn as nn
import torch.nn.functional as F

from nets_nano_cnn.img_branch.ds_conv_img import ds_basic_conv


class basic_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(basic_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_left = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                   padding=0)
        self.norm_left = nn.BatchNorm2d(in_channels)

        self.conv_center = ds_basic_conv(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1,
                                         padding=1)
        self.norm_center = nn.BatchNorm2d(in_channels)

        self.norm_right = nn.BatchNorm2d(in_channels)
        # ================================================================== #
        self.act_center = nn.LeakyReLU(inplace=False)
        # ================================================================== #
        self.conv_center_next = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2,
                                          padding=1)
        self.norm_center_next = nn.BatchNorm2d(out_channels)
        self.act_center_last = nn.LeakyReLU(inplace=False)
        # ================================================================== #
        self.conv_res = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2,
                                          padding=1)
        self.norm_res = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_left, x_center, x_right, x_res = x, x, x, x
        # === left ===============================
        x_left = self.conv_left(x_left)
        x_left = self.norm_left(x_left)

        # === center =============================
        x_center = self.conv_center(x_center)
        x_center = self.norm_center(x_center)

        # === right ==============================
        x_right = self.norm_right(x_right)

        x_center = x_left + x_center + x_right
        x_center = self.act_center(x_center)

        # ================= stage2 ===================
        x_center = self.conv_center_next(x_center)
        x_center = self.norm_center_next(x_center)

        x_res = self.conv_res(x_res)
        x_res = self.norm_res(x_res)

        x_center = x_center + x_res
        x_center = self.act_center_last(x_center)

        return x_center


class img_encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(img_encoder, self).__init__()
        self.in_channels = in_channels
        self.conv_stage1 = basic_conv(in_channels=in_channels, out_channels=32)
        self.conv_stage2 = basic_conv(in_channels=32, out_channels=64)
        self.conv_stage3 = basic_conv(in_channels=64, out_channels=128)
        self.conv_stage4 = basic_conv(in_channels=128, out_channels=256)
        self.conv_stage5 = basic_conv(in_channels=256, out_channels=512)

    def forward(self, x):
        x = self.conv_stage1(x)
        x = self.conv_stage2(x)
        x = self.conv_stage3(x)
        x = self.conv_stage4(x)
        x = self.conv_stage5(x)
        return x



if __name__ == '__main__':
    test_map = torch.randn(16, 3, 384, 384)
    bs = img_encoder(in_channels=3)
    output_map = bs(test_map)
    print(output_map.shape)


