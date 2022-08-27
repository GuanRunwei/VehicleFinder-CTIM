import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import trange
from utils.dataset import get_dataloader
from torch.utils.tensorboard import SummaryWriter
from utils.callbacks import loss_save
import datetime
import random
import os
from utils.dataset import get_dataloader
from nets_nano_cnn.vehiclefinder import vehicle_finder_siamese

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def get_accuracy(prediction, gt):
    prediction_results = []
    for item in prediction:
        if item[0] >= 0.5:
            prediction_results.append(1)
        elif item[0] < 0.5:
            prediction_results.append(0)

    gt = gt.squeeze(1)
    gt = [int(item) for item in gt]
    prediction_results = np.array(prediction_results)
    # print(np.sum(prediction == gt) / len(gt))
    return np.sum(prediction_results == gt) / len(gt)




if __name__ == '__main__':
    # -------------------------------- 超参数 -------------------------------- #
    batch_size = 64
    train_ratio = 0.7
    cuda = True
    mix_precision = False
    optimizer_name = 'adamW'
    scheduler_name = 'step'
    learning_rate = 0.001
    weight_decay = 5e-4
    epochs = 100
    criterion = nn.BCELoss()
    # ------------------------------------------------------------------------ #

    # ------------------------------- 训练设备 --------------------------------- #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # -------------------------------------------------------------------------- #

    # --------------------------------- SEED ------------------------------------- #
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    setup_seed(777)
    # ---------------------------------------------------------------------------- #

    # ------------------------------- 数据集加载 ---------------------------------- #
    trainloader, testloader, validloader = get_dataloader(train_ratio=train_ratio, batch_size=batch_size)
    # ----------------------------------------------------------------------------- #

    # ------------------------------- 模型加载 ------------------------------------ #
    vehicle_finder_model = vehicle_finder_siamese(img_channels=3, text_channels=1).to(device)

    vehicle_finder_model = nn.DataParallel(vehicle_finder_model, device_ids=[0, 1])
    # ----------------------------------------------------------------------------- #

    # ------------------------------ Optimizer --------------------------------- #
    if optimizer_name == 'adamW':
        optimizer = optim.AdamW(lr=learning_rate, params=vehicle_finder_model.parameters(), weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(lr=learning_rate, params=vehicle_finder_model.parameters(), momentum=0.937)
    # -------------------------------------------------------------------------- #


    # ------------------------------ Scheduler --------------------------------- #
    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, eta_min=learning_rate * 0.05, T_max=epochs/10)
    elif scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.9, step_size=1)
    # -------------------------------------------------------------------------- #


    # ------------------------------ Start Training ---------------------------- #
    print()
    print("================= Training Configuration ===================")
    print("trainloader size:", len(trainloader) * batch_size)
    print("validloader size:", len(validloader) * batch_size)
    print("testloader size:", len(testloader) * batch_size)
    print("epoch:", epochs)
    print("batch size:", batch_size)
    print("optimizer:", optimizer_name)
    print("scheduler:", scheduler_name)
    print("initial learning rate:", learning_rate)
    print("weight decay:", weight_decay)
    print("device:", device)
    print("mix precision:", mix_precision)
    print("=============================================================")
    mse_loss_min = 1000000
    train_loss_array = []
    valid_loss_array = []
    best_model = None
    best_model_name = None
    for epoch in range(epochs):
        train_loss = 0
        train_loop = tqdm(enumerate(trainloader), total=len(trainloader))
        vehicle_finder_model.train()
        for i, (words, images, labels) in train_loop:
            words_input = words.to(device)
            images_input = images.to(device)
            gts = labels.to(device)
            predictions = vehicle_finder_model(images_input, words_input).unsqueeze(-1).to(device)
            train_accuracy = get_accuracy(predictions, gts)
            loss = criterion(predictions, gts)
            train_loss += loss.item()

            # ------------------ 清空梯度,反向传播 ----------------- #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ----------------------------------------------------- #
            train_loop.set_description(f'Epoch [{epoch}/{epochs}]')
            train_loop.set_postfix(BCE_Loss=loss.item(), train_accu=train_accuracy,
                                   learning_rate=optimizer.param_groups[0]['lr'])
        train_loss_array.append(train_loss)

        # log_history.write(str(loss.item()) + '\n')

        # loss_history = LossHistory(log_dir="logs", model=nano_sta_model, input_shape=[128, 64])

        # ------------------------------- Validation --------------------------------- #
        validation_loop = tqdm(enumerate(validloader), total=len(validloader))

        print()
        print("########################## start validation #############################")
        vehicle_finder_model.eval()
        with torch.no_grad():
            validation_loss = 0
            for i, (words, images, labels) in validation_loop:
                words_input = words.to(device)
                images_input = images.to(device)
                gts = labels.to(device)
                predictions = vehicle_finder_model(images_input, words_input).unsqueeze(-1).to(device)
                loss = criterion(predictions, gts)
                valid_accuracy = get_accuracy(predictions, gts)
                validation_loss += loss.item()
                validation_loop.set_postfix(loss_real_time=loss.item(), validation_accu=valid_accuracy)
            if validation_loss < mse_loss_min:
                best_model = vehicle_finder_model
                best_model_name = "val_loss_" + str(validation_loss) + '.pth'
                print("best model now:", best_model_name)
                torch.save(best_model.state_dict(), best_model_name)
                mse_loss_min = validation_loss
            # writer.add_scalar("MSE - Validation", validation_loss / (len(validloader) * batch_size), epoch)
            # if validation_loss / (len(validloader) * batch_size) <= mse_loss_min:
            #     torch.save(nano_sta_model.state_dict(), "logs/validation " +
            #                str(validation_loss / (len(validloader) * batch_size)) + ".pth")
        valid_loss_array.append(validation_loss)

        print()
        print("########################## end validation #############################")
        # ---------------------------------------------------------------------------- #

        scheduler.step()

    loss_save(train_loss_array, mode='train')
    loss_save(valid_loss_array, mode='valid', model=best_model, model_name=best_model_name)
    print()
    print("============================== end training =================================")




    