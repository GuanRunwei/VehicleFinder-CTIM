import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from nets_nano_cnn.text_branch.text_encoder import text_encoder
from nets_nano_cnn.img_branch.img_encoder import img_encoder
from thop import profile
from thop import clever_format
from einops import rearrange

class Cigmoid(nn.Module):
    def forward(self, x):
        x = (1+x) / (1 + torch.exp(-x+1))
        return x

class Cigmoid2(nn.Module):
    def forward(self, x):
        x = 0.5*x + 0.5
        return x


##  ============ CTIM =============
class vehicle_finder_siamese(nn.Module):
    def __init__(self, img_channels, text_channels):
        super(vehicle_finder_siamese, self).__init__()
        self.img_channels = img_channels
        self.text_channels = text_channels
        self.flatten_shape = 512 * 288
        self.img_encoder = img_encoder(in_channels=img_channels)
        self.text_encoder = text_encoder(in_channels=text_channels)
        self.activation = Cigmoid2()

        # self.fc1 = nn.Linear(73728, 1024)
        # self.drop = nn.Dropout(0.05)
        # self.fc2 = nn.Linear(1024, 1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, img, text):
        img_vector = self.img_encoder(img)

        img_vector = img_vector.flatten(1)
        text_vector = self.text_encoder(text).flatten(1)
        # text_vector = rearrange(text_vector, 'b c (h w) -> b c h w',
        #                         h=int(math.sqrt(text_vector.shape[2])),
        #                         w=int(math.sqrt(text_vector.shape[2])))


        output = torch.cosine_similarity(img_vector, text_vector).cuda()
        output = self.activation(output)
        return output


if __name__ == '__main__':
    img_vector = torch.randn(16, 3, 384, 384).cuda()
    text_vector = torch.randn(16, 1, 3, 300).cuda()
    vs = vehicle_finder_siamese(3, 1).cuda()
    output = vs(img_vector, text_vector)
    macs, params = profile(vs, inputs=(img_vector, text_vector))
    macs, params = clever_format([macs, params], "%.3f")
    print("========== clever mode ==========")
    print("macs:", macs)
    print("params:", params)
    latency, fps = get_fps(input=input, model=vs, test_times=100)
    print("FPS:", fps)
    print("latency:", latency * 1000, " ms")
