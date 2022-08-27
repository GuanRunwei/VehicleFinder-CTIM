import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import os
from utils.image_transforms import image_transforms
import jieba
import cv2


class CLIP_Dataset(Dataset):
    def __init__(self, pickle_path="E:/Big_Datasets/Vehicle_Detection/DETRAC/CLIP_Project/multi_label_proposal_annotation_new.pkl",
                 word_vector_path="E:/Big_Datasets/Vehicle_Detection/fasttext_pretained/wiki-news-300d-1M.vec/wiki-news-300d-1M.vec",
                 image_root_path="E:/Big_Datasets/Vehicle_Detection/DETRAC/CLIP_Project/proposals",
                 transforms=image_transforms):
        self.pickle_path = pickle_path
        self.word_vector_path = word_vector_path
        self.image_root_path = image_root_path
        self.word_vector_model = KeyedVectors.load_word2vec_format(self.word_vector_path, limit=500000)
        self.dataset = pd.read_pickle(self.pickle_path)
        self.transforms = transforms

    def _read_rgb_image(self, path):
        dim = (384, 384)
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float64)
        image = cv2.resize(image, dim)
        image = np.transpose(image, (2, 0, 1))
        image /= 255.0
        return image

    def _segment_words(self, words: str):
        sub_words = words.split(' ')
        sub_words_count = len(sub_words)
        return sub_words, sub_words_count

    def _get_words_vector(self, words):
        sub_words, sub_words_count = self._segment_words(words)
        words_vector = []
        for sub_word in sub_words:
            words_vector.append(self.word_vector_model[sub_word])
        return np.mean(words_vector, axis=0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_item = self.dataset.loc[index]
        vehicle = data_item['vehicle']
        color = data_item['color']
        orientation = data_item['orientation']
        if self._segment_words(vehicle)[1] > 1:
            vehicle_vector = self._get_words_vector(vehicle)
        else:
            vehicle_vector = self.word_vector_model[vehicle]

        if self._segment_words(color)[1] > 1:
            color_vector = self._get_words_vector(color)
        else:
            color_vector = self.word_vector_model[color]

        if self._segment_words(orientation)[1] > 1:
            orientation_vector = self._get_words_vector(orientation)
        else:
            orientation_vector = self.word_vector_model[orientation]

        image_path = os.path.join(self.image_root_path, data_item['proposal_path']+'.png')
        label = torch.as_tensor(data_item['label'])

        word_vector_matrix = np.array([[vehicle_vector, color_vector, orientation_vector]])

        image_matrix = self.transforms(self._read_rgb_image(image_path))

        return word_vector_matrix, image_matrix, label


def dataset_collate(batch):
    words = []
    images = []
    labels = []
    for word_matrix, image_matrix, label in batch:
        words.append(word_matrix)
        images.append(image_matrix)
        labels.append([label])

    words_results = torch.from_numpy(np.array(words, dtype=np.float64)).type(torch.FloatTensor)
    images_results = torch.from_numpy(np.array(images, dtype=np.float64)).type(torch.FloatTensor)
    labels_results = torch.from_numpy(np.array(labels, dtype=np.float64)).type(torch.FloatTensor)

    return words_results, images_results, labels_results


def get_dataloader(train_ratio=0.8, batch_size=8):
    dataset = CLIP_Dataset()

    # --------------- 设置训练、测试、验证数据索引，注意Sampler默认shuffle, Dataloader无需shuffle，否则报错 ----------------- #
    datalen = dataset.__len__()
    dataidx = np.array(list(range(datalen)))
    np.random.shuffle(dataidx)

    splitfrac = train_ratio
    split_idx = int(splitfrac * datalen)
    train_idxs = dataidx[:split_idx]
    valid_idxs = dataidx[split_idx:]

    testsplit = 0.2
    testidxs = int(testsplit * len(train_idxs))

    test_idxs = train_idxs[:testidxs]
    train_idxs = train_idxs[testidxs:]

    np.random.shuffle(test_idxs)

    train_samples = torch.utils.data.SubsetRandomSampler(train_idxs)
    valid_samples = torch.utils.data.SubsetRandomSampler(valid_idxs)
    test_samples = torch.utils.data.SubsetRandomSampler(test_idxs)
    # print(len(train_samples))
    # print(len(valid_samples))
    # print(len(test_samples))
    # ------------------------------------------------------------- #

    # --------------- 训练、测试、验证dataloader ----------------- #
    trainloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_samples, collate_fn=dataset_collate, num_workers=4)
    testloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=test_samples, collate_fn=dataset_collate, num_workers=4)
    validloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=valid_samples, collate_fn=dataset_collate, num_workers=4)

    return trainloader, testloader, validloader







if __name__ == '__main__':
    # data = CLIP_Dataset()
    # print(data[0])
    trainloader, testloader, validloader = get_dataloader(batch_size=16,
                                                          train_ratio=0.9)
    print("trainloader size:", len(trainloader))
    print("validloader size:", len(validloader))
    print("testloader size:", len(testloader))

    word, image, label = next(iter(trainloader))
    print("word.shape:", word.shape)
    print("image.shape:", image.shape)
    print("label.shape:", label.shape)



