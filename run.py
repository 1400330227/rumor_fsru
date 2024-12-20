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

    valid_acc_sum, valid_pre_sum, valid_recall_sum, valid_f1_sum = 0., 0., 0., 0.
    valid_nr_pre_sum, valid_nr_recall_sum, valid_nr_f1_sum = 0., 0., 0.
    valid_r_pre_sum, valid_r_recall_sum, valid_r_f1_sum = 0., 0., 0.
    train, valid = {}, {}

    (train['text'], train['image'], train['label'],
     valid['text'], valid['image'], valid['label']) = get_kfold_data(text, image, label)


    train_loader = DataLoader(dataset=MyDataset(train), batch_size=args.batch_size, shuffle=False)
    valid_loader = DataLoader(dataset=MyDataset(valid), batch_size=args.batch_size, shuffle=False)

    print('Building model...')

    model = FSRU(W, args.vocab_size, args.d_text, args.seq_len, args.img_size, args.patch_size, args.d_model,
                 args.num_filter, args.num_class, args.num_layer, args.dropout)
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)

    best_valid_acc, best_valid_pre, best_valid_recall, best_valid_f1 = 0., 0., 0., 0.
    best_valid_nr_pre, best_valid_nr_f1, best_valid_nr_recall, = 0., 0., 0.
    best_valid_r_pre, best_valid_r_recall, best_valid_r_f1 = 0., 0., 0.
    loss_list = []
    acc_list = []
    for epoch in range(args.num_epoch):
        train_losses, valid_losses, train_acc, valid_acc = [], [], [], []
        start_time = time.time()
        cls_loss = []

        # train
        model.train()
        train_pred, train_y = [], []
        for j, (train_text, train_image, train_labels) in enumerate(train_loader):
            num_r, num_nr = count(train_labels)
            train_text, train_image, train_labels = to_var(train_text), to_var(train_image), to_var(train_labels)

            criterion_full = FullContrastiveLoss(batch_size=train_text.shape[0], num_r=num_r, num_nr=num_nr)
            criterion_self = SelfContrastiveLoss(batch_size=train_text.shape[0])
            optimizer.zero_grad()

            text_outputs, image_outputs, label_outputs, _ = model(train_text, train_image)

            loss = criterion(label_outputs, train_labels.long())
            loss_full = criterion_full(text_outputs, image_outputs, train_labels.long())
            loss_self = criterion_self(text_outputs, image_outputs)
            train_loss = loss + args.alpha * loss_full + args.beta * loss_self
            train_loss.backward()
            optimizer.step()
            pred = torch.max(label_outputs, 1)[1]

            if j == 0:
                train_pred = to_np(pred.squeeze())
                train_y = to_np(train_labels.squeeze())
            else:
                train_pred = np.concatenate((train_pred, to_np(pred.squeeze())), axis=0)
                train_y = np.concatenate((train_y, to_np(train_labels.squeeze())), axis=0)

            train_accuracy = torch.eq(train_labels, pred.squeeze()).float().mean()  # .sum() / len(train_labels)
            train_losses.append(train_loss.item())
            train_acc.append(train_accuracy.item())
            cls_loss.append(loss.item())

        if epoch % args.decay_step == 0:
            for params in optimizer.param_groups:
                params['lr'] *= args.decay_rate

        # valid
        model.eval()
        valid_pred, valid_y = [], []
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

        cur_valid_acc = metrics.accuracy_score(valid_y, valid_pred)
        valid_pre = metrics.precision_score(valid_y, valid_pred, average='macro')
        valid_recall = metrics.recall_score(valid_y, valid_pred, average='macro')
        valid_f1 = metrics.f1_score(valid_y, valid_pred, average='macro')
        duration = time.time() - start_time
        print('Epoch[{}/{}], Duration:{:.8f}, Loss:{:.8f}, Train_Accuracy:{:.5f}, Valid_accuracy:{:.5f}, Len_Train: {:.1f}'.format(
            epoch + 1, args.num_epoch, duration, np.mean(train_losses), metrics.accuracy_score(train_y, train_pred), cur_valid_acc, len(train_y)))
        loss_list.append(np.mean(cls_loss))
        acc_list.append(cur_valid_acc)

        if cur_valid_acc > best_valid_acc:
            best_valid_acc = cur_valid_acc
            best_valid_pre = valid_pre
            best_valid_recall = valid_recall
            best_valid_f1 = valid_f1
            print('Best...')

            target_names = ['non-rumor', 'rumor']
            report = metrics.classification_report(valid_y, valid_pred, output_dict=True, target_names=target_names)
            nr_report = report['non-rumor']
            best_valid_nr_pre = nr_report['precision']
            best_valid_nr_recall = nr_report['recall']
            best_valid_nr_f1 = nr_report['f1-score']
            r_report = report['rumor']
            best_valid_r_pre = r_report['precision']
            best_valid_r_recall = r_report['recall']
            best_valid_r_f1 = r_report['f1-score']

    torch.save(model, './models/best_model.pth')
    valid_acc_sum += best_valid_acc
    valid_pre_sum += best_valid_pre
    valid_recall_sum += best_valid_recall
    valid_f1_sum += best_valid_f1
    print('best_valid_acc:{:.6f}, best_valid_pre:{:.6f}, best_valid_recall:{:.6f}, best_valid_f1:{:.6f}'.
          format(best_valid_acc, best_valid_pre, best_valid_recall, best_valid_f1))
    valid_nr_pre_sum += best_valid_nr_pre
    valid_nr_recall_sum += best_valid_nr_recall
    valid_nr_f1_sum += best_valid_nr_f1
    valid_r_pre_sum += best_valid_r_pre
    valid_r_recall_sum += best_valid_r_recall
    valid_r_f1_sum += best_valid_r_f1

    print('=' * 40)
    print('Accuracy:{:.5f}, F1:{:.5f}'.format(valid_acc_sum, valid_f1_sum))
    print('Rumor Precision:{:.5f}, Rumor Recall:{:.5f}, Rumor F1:{:.5f}'.format(
        valid_r_pre_sum, valid_r_recall_sum, valid_r_f1_sum))
    print('Non-Rumor Precision:{:.5f}, Non-Rumor Recall:{:.5f}, Non-Rumor F1:{:.5f}'.format(
        valid_nr_pre_sum, valid_nr_recall_sum, valid_nr_f1_sum))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    args = parser.parse_args()
    main(args)
