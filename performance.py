import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import trange
from nets_nano_cnn.vehiclefinder import vehicle_finder_siamese
from torch.utils.tensorboard import SummaryWriter
from utils.callbacks import loss_save
import datetime
import time
import matplotlib.pyplot as plt
import random
import os
from thop import profile
from thop import clever_format


def get_fps(img_input, text_input, model, test_times=300):
    t1 = time.time()

    for _ in range(test_times):
        output = model(img_input, text_input)

    t2 = time.time()

    return ((t2 - t1) / test_times), 1 / ((t2 - t1) / test_times)






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
    latency, fps = get_fps(img_input=img_vector, text_input=text_vector, model=vs, test_times=100)
    print("FPS:", fps)
    print("latency:", latency * 1000, " ms")
