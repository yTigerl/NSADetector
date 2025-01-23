import random

import numpy as np
import pytorch_lightning as pl
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
# from learningSandwich.util import *
import numpy as np

num = 80000
featureNum = 11

class SiameseDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        img1, label1 = self.dataset[index]
        label2 = label1
        while label2 == label1:
            index2 = torch.randint(0, len(self.dataset), (1,)).item()
            img2, label2 = self.dataset[index2]
        return img1, img2, torch.tensor(int(label1 != label2), dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)

def read_data_num(feature_file, label_file, num):
    with open(feature_file, 'r') as f:
        feature_lines = f.readlines()

    with open(label_file, 'r') as f:
        label_lines = f.readlines()

    data = []
    target = []
    current_sample = []
    # print(feature_lines)
    sample = 0
    counts = [0, 0, 0, 0, 0]
    for feature_line in feature_lines:
        if feature_line.strip():  # 跳过空行
            # print(feature_line)
            current_sample.append(np.array(list(map(float, feature_line.strip().split()))))
        else:
            if counts[int(label_lines[sample].strip())] < num:
                # print(sample)
                # print(len(current_sample))
                # print(np.array(current_sample).size)
                if np.array(current_sample).size != 7*5*featureNum:
                    # print(np.array(current_sample))
                    sample += 1
                    current_sample = []
                    continue
                # print(np.array(current_sample).reshape((1, 7 * 5, 8)))
                data.append(np.array(current_sample).reshape((1, 7 * 5, featureNum)))
                # print(data.__sizeof__())
                target.append(int(label_lines[sample].strip()))
                counts[int(label_lines[sample].strip())] += 1
            current_sample = []
            sample += 1

    if sample < len(label_lines) and counts[int(label_lines[sample].strip())] < num and np.array(current_sample).size == 7*5*8:

        data.append(np.array(current_sample).reshape((1, 35, 8)))
        current_sample = []
        target.append(int(label_lines[sample].strip()))
        counts[int(label_lines[sample].strip())] += 1

    return np.array(data), np.array(target)

def process_data(feature_file, label_file, num_classes, classes):
    with open(feature_file, 'r') as file:
        data = file.read()
    with open(label_file, 'r') as f:
        label_lines = f.readlines()
    # Split data into samples using double newline as the delimiter
    raw_samples = data.strip().split('\n\n')

    samples = []
    fromto = []
    types = []
    pairaddress = []
    amount = []
    labels = []
    target = []
    current_sample = []
    # print(feature_lines)
    count = 0
    counts = [0, 0, 0, 0]
    for raw_sample in raw_samples:
        fromto = []
        types = []
        pairaddress = []
        amount = []
        sample = []
        lines = raw_sample.split('\n')
        oldvalues = None
        for line in lines:
            # Split line by commas and convert each element to float
            values = list(map(float, line.split(' ')))
            sample.append(values)
            if values[0] != -1:
                oldvalues = values
                types.append(values[7])
                pairaddress.append(values[2])
                amount.append(values[3])
                amount.append(values[4])
                amount.append(values[5])
                amount.append(values[6])
            else:
                fromto.append(oldvalues[0])
                fromto.append(-1)
                types.append(-1)
                pairaddress.append(-1)
                amount.append(-1)
        if len(fromto) < 30:
            fromto = np.hstack((fromto, [-1]*(30-len(fromto))))
        else:
            fromto = fromto[:30]
        if len(types) < 20:
            types = np.hstack((types, [-1]*(20-len(types))))
        else:
            types = types[:20]
        if len(pairaddress) < 20:
            pairaddress = np.hstack((pairaddress, [-1]*(20-len(pairaddress))))
        else:
            pairaddress = pairaddress[:20]
        if len(amount) < 50:
            amount = np.hstack((amount, [-1]*(50-len(amount))))
        else:
            amount = amount[:50]
        # samples.append(sample)
        data = np.hstack((fromto, types, pairaddress, amount))
        target.append(int(label_lines[count].strip()))
        count += 1
        samples.append(data)
    # for sample in samples:
    category = [np.where(np.array(target) == classes[i])[0] for i in range(num_classes)]
    # print(category.shape)
    tr_pairs, tr_y = create_pairs(np.array(samples), category, num_classes)
    return tr_pairs, tr_y
    # return samples

def load_dataset(dataset):
    data = []
    labels = []
    for i in range(len(dataset)):
        data.append(dataset[i][0].numpy())
        labels.append(dataset[i][1])
    return data, labels

def save_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    # train_dataset = SiameseDataset(train_dataset)
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_data, train_labels = load_dataset(train_dataset)
    test_data, test_labels = load_dataset(val_dataset)
    # np.savetxt('data/train_data.txt', train_data,fmt='%0.8f')
    np.save(file="data/train_data.txt", arr=train_data)
    np.savetxt('data/train_label.txt', train_labels,fmt='%0.8f')
    np.save(file="data/test_data.txt", arr=test_data)
    # np.savetxt('data/test_data.txt',test_data,fmt='%0.8f')
    np.savetxt('data/test_label.txt', test_labels,fmt='%0.8f')
    # print(train_data)
    # print(train_labels)


# print(category)

def create_pairs(x, category, num):
    pairs = []
    labels = []

    # n: the least number of samples in all classes. (The numbers of samples in each class are different. Thus we use min() to guarantee pairs.)
    # note that category has shape (num_classes, num_samples)
    n = min([len(category[d]) for d in range(num)]) - 1
    print(n)
    for d in range(num):
        for i in range(n):
            # create positive pairs
            z1, z2 = category[d][i], category[d][i + 1]  # indexes of the same class
            pairs += [[x[z1], x[z2]]]  # '+=' operation is equivalent to .extend([a, b])

            # create negative pairs
            inc = random.randint(1, num)  # return a randomly selected element a, 1 <= a < num
            dn = (d + inc) % num  # This func guaranteed that dn != d
            z1, z2 = category[d][i], category[dn][i]
            pairs += [[x[z1], x[z2]]]  # negative pairs
            labels += [1, 0]  # '+=' operation is equivalent to .extend([a, b])

    # convert the list to numpy array
    return np.array(pairs), np.array(labels)



def load_siameseTrainDataset():
    train_data = np.load(file="data/train_data.txt.npy")
    train_labels = np.loadtxt('data/train_label.txt')
    # print(train_data[0])
    # print(train_labels)

    # b=np.loadtxt('a.txt',dtype=np.float32)
    num_classes = 10
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    category = [np.where(train_labels == classes[i])[0] for i in range(num_classes)]
    tr_pairs, tr_y = create_pairs(train_data, category, num_classes)
    return tr_pairs, tr_y

def load_siameseTestDataset():
    test_data = np.load(file="data/test_data.txt.npy")
    test_labels = np.loadtxt('data/test_label.txt')
    # print(train_data[0])
    # print(train_labels)

    # b=np.loadtxt('a.txt',dtype=np.float32)
    num_classes = 10
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    category_test = [np.where(test_labels == classes[j])[0] for j in range(num_classes)]
    te_pairs, te_y = create_pairs(test_data, category_test, num_classes)
    return te_pairs, te_y


def load_sandwichTrainDataset(feature_file, label_file, num_classes, classes):
    train_data, train_labels = read_data_num(feature_file, label_file, num)
    print(train_data[0])
    print(train_data.shape)
    # print(len(train_labels))
    # num_classes = 4
    # classes = [0, 1, 2, 3]

    category = [np.where(train_labels == classes[i])[0] for i in range(num_classes)]
    # print(category.shape)
    tr_pairs, tr_y = create_pairs(train_data, category, num_classes)
    return tr_pairs, tr_y

def load_sandwichTestDataset(feature_file, label_file, num_classes, classes):
    test_data, test_labels = read_data_num(feature_file, label_file, num)
    # num_classes = 4
    # classes = [0, 1, 2, 3]
    category_test = [np.where(test_labels == classes[j])[0] for j in range(num_classes)]
    te_pairs, te_y = create_pairs(test_data, category_test, num_classes)
    return te_pairs, te_y

def splitDataset(feature_file, label_file, num_classes, classes):
    # input_data, target_data = read_data_num(feature_file, label_file, num)
    #
    # sample_features_tensor = torch.tensor(input_data, dtype=torch.float32)
    # labels_tensor = torch.tensor(target_data, dtype=torch.int64)
    #
    # # 创建 TensorDataset
    # dataset = TensorDataset(sample_features_tensor, labels_tensor)
    #
    # # 切分训练集、测试集和验证集（80%训练集，10%测试集，10%验证集）
    # train_size = int(0.8 * len(dataset))
    # test_val_size = len(dataset) - train_size
    # test_size = int(0.5 * test_val_size)
    # val_size = test_val_size - test_size
    #
    # train_dataset, test_val_dataset = torch.utils.data.random_split(dataset, [train_size, test_val_size])
    # test_dataset, val_dataset = torch.utils.data.random_split(test_val_dataset, [test_size, val_size])

    tr_pairs, tr_y = load_sandwichTrainDataset(feature_file, label_file, num_classes, classes)
    # print(tr_y.shape)

    tr_pairs_0 = np.vstack((tr_pairs[0:1336, 0], tr_pairs[1670:3006, 0], tr_pairs[3340:4676, 0], tr_pairs[5010:6346, 0]))
    tr_pairs_1 = np.vstack((tr_pairs[0:1336, 1], tr_pairs[1670:3006, 1], tr_pairs[3340:4676, 1], tr_pairs[5010:6346, 1]))
    tr_y_ = np.hstack((tr_y[0:1336], tr_y[1670:3006], tr_y[3340:4676], tr_y[5010:6346]))
    # print(tr_y_.shape)
    # tr_y_ =
    test_pairs_0 = np.vstack(
        (tr_pairs[1336:1502, 0], tr_pairs[3006:3172, 0], tr_pairs[4676:4842, 0], tr_pairs[6346:6512, 0]))
    test_pairs_1 = np.vstack(
        (tr_pairs[1336:1502, 1], tr_pairs[3006:3172, 1], tr_pairs[4676:4842, 1], tr_pairs[6346:6512, 1]))
    test_y_ = np.hstack((tr_y[1336:1502], tr_y[3006:3172], tr_y[4676:4842], tr_y[6346:6512]))
    val_pairs_0 = np.vstack(
        (tr_pairs[1502:1670, 0], tr_pairs[3172:3340, 0], tr_pairs[4842:5010, 0], tr_pairs[6512:6680, 0]))
    val_pairs_1 = np.vstack(
        (tr_pairs[1502:1670, 1], tr_pairs[3172:3340, 1], tr_pairs[4842:5010, 1], tr_pairs[6512:6680, 1]))
    val_y_ = np.hstack((tr_y[1502:1670], tr_y[3172:3340], tr_y[4842:5010], tr_y[6512:6680]))
    tr_pairs_0 = torch.tensor(tr_pairs_0, dtype=torch.float32)
    tr_pairs_1 = torch.tensor(tr_pairs_1, dtype=torch.float32)
    tr_y_ = torch.tensor(tr_y_, dtype=torch.int64)
    test_pairs_0 = torch.tensor(test_pairs_0, dtype=torch.float32)
    test_pairs_1 = torch.tensor(test_pairs_1, dtype=torch.float32)
    test_y_ = torch.tensor(test_y_, dtype=torch.int64)
    val_pairs_0 = torch.tensor(val_pairs_0, dtype=torch.float32)
    val_pairs_1 = torch.tensor(val_pairs_1, dtype=torch.float32)
    val_y_ = torch.tensor(val_y_, dtype=torch.int64)
    train_dataset = TensorDataset(tr_pairs_0, tr_pairs_1, tr_y_)
    test_dataset = TensorDataset(test_pairs_0, test_pairs_1, test_y_)
    val_dataset = TensorDataset(val_pairs_0, val_pairs_1, val_y_)

    # tr_pairs = torch.tensor(tr_pairs, dtype=torch.float32)
    # tr_y = torch.tensor(tr_y, dtype=torch.int64)
    # # dataset = TensorDataset(tr_pairs[: , 0], tr_pairs[: , 1], tr_y)
    #
    # train_dataset = TensorDataset(tr_pairs[0:1336, 0] + tr_pairs[1670:3006, 0] + tr_pairs[3340:4676, 0] + tr_pairs[5010:6346, 0],
    #                               tr_pairs[0:1336, 1] + tr_pairs[1670:3006, 1] + tr_pairs[3340:4676, 1] + tr_pairs[5010:6346, 1],
    #                               tr_y[0:1336] + tr_y[1670:3006] + tr_y[3340:4676] + tr_y[5010:6346])
    # test_dataset = TensorDataset(
    #     tr_pairs[1336:1502, 0] + tr_pairs[3006:3172, 0] + tr_pairs[4676:4842, 0] + tr_pairs[6346:6512, 0],
    #     tr_pairs[1336:1502, 1] + tr_pairs[3006:3172, 1] + tr_pairs[4676:4842, 1] + tr_pairs[6346:6512, 1],
    #     tr_y[1336:1502] + tr_y[3006:3172] + tr_y[4676:4842] + tr_y[6346:6512])
    # val_dataset = TensorDataset(
    #     tr_pairs[1502:1670, 0] + tr_pairs[3172:3340, 0] + tr_pairs[4842:5010, 0] + tr_pairs[6512:6680, 0],
    #     tr_pairs[1502:1670, 1] + tr_pairs[3172:3340, 1] + tr_pairs[4842:5010, 1] + tr_pairs[6512:6680, 1],
    #     tr_y[1502:1670] + tr_y[3172:3340] + tr_y[4842:5010] + tr_y[6512:6680])
    # print(dataset[0])
    # train_size = int(0.8 * len(dataset))
    # test_val_size = len(dataset) - train_size
    # test_size = int(0.5 * test_val_size)
    # val_size = test_val_size - test_size
    #
    # train_dataset, test_val_dataset = torch.utils.data.random_split(dataset, [train_size, test_val_size])
    # test_dataset, val_dataset = torch.utils.data.random_split(test_val_dataset, [test_size, val_size])
    return train_dataset, test_dataset, val_dataset

