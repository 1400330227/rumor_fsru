import argparse
import random
import math
import numpy as np
import os
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.metrics import classification_report
from arguments import parse_arguments
from data_loader import *
from datasets import MyDataset
from model import FSRU
from loss import FullContrastiveLoss, SelfContrastiveLoss

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.manual_seed(30)
torch.manual_seed(30)
np.random.seed(30)
random.seed(30)


def to_var(x):
    if torch.cuda.is_available():
        x = torch.as_tensor(x, dtype=torch.float32).cuda()
    else:
        x = torch.as_tensor(x, dtype=torch.float32)
    return x


def to_np(x):
    return x.data.cpu().numpy()


def get_kfold_data(text, image, label):
    fold_size = text.shape[0]
    val_start = int(fold_size * 0.8)

    text_train, image_train, label_train = text[0:val_start], image[0:val_start], label[0:val_start]
    text_valid, image_valid, label_valid = text[val_start:], image[val_start:], label[val_start:]

    return text_train, image_train, label_train, text_valid, image_valid, label_valid


def count(labels):
    r, nr = 0, 0
    for label in labels:
        if label == 0:
            nr += 1
        elif label == 1:
            r += 1
    return r, nr


def shuffle_dataset(text, image, label):
    assert len(text) == len(image) == len(label)
    rp = np.random.permutation(len(text))
    text = text[rp]
    image = image[rp]
    label = label[rp]

    return text, image, label


def main(args):
    print('Loading data ...')

    text, image, label, W = load_data(args)
    text, image, label = shuffle_dataset(text, image, label)
    train, valid = {}, {}

    (train['text'], train['image'], train['label'],
     valid['text'], valid['image'], valid['label']) = get_kfold_data(text, image, label)

    valid_loader = DataLoader(dataset=MyDataset(valid), batch_size=args.batch_size, shuffle=False)

    print('Building model...')
    model = torch.load('./models/best_model.pth')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for j, (valid_text, valid_image, valid_labels) in enumerate(valid_loader):
            valid_text, valid_image, valid_labels = to_var(valid_text), to_var(valid_image), to_var(
                valid_labels)
            _, _, label_outputs, _ = model(valid_text, valid_image)
            label_outputs = F.softmax(label_outputs, dim=1)
            pred = torch.max(label_outputs, 1)[1]
            if j == 0:
                valid_pred = to_np(pred.squeeze())
                valid_y = to_np(valid_labels.squeeze())
            else:
                valid_pred = np.concatenate((valid_pred, to_np(pred.squeeze())), axis=0)
                valid_y = np.concatenate((valid_y, to_np(valid_labels.squeeze())), axis=0)
        result = (valid_y == valid_pred).sum()
        lens = len(valid_loader.dataset)
        print('正确数:{:.2f}, 总数:{:.2f}'.format(result, lens))
        valid_acc = metrics.accuracy_score(valid_y, valid_pred)
        valid_pre = metrics.precision_score(valid_y, valid_pred, average='macro')
        valid_recall = metrics.recall_score(valid_y, valid_pred, average='macro')
        valid_f1 = metrics.f1_score(valid_y, valid_pred, average='macro')

        print('在测试集上的准确率为:{:.2f}%'.format(100. * valid_acc))
        print('在测试集上的精确率为:{:.2f}%'.format(100. * valid_pre))
        print('在测试集上的召回率为:{:.2f}%'.format(100. * valid_recall))
        print('在测试集上的F1为:{:.2f}%'.format(100. * valid_f1))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    args = parser.parse_args()
    main(args)
