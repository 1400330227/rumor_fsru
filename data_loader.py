"""
An Lao
"""
import os
import copy
import numpy as np
import pickle
import pandas as pd
import torch
from PIL import Image
from collections import defaultdict
import torch.nn.functional as F
from functorch.dim import Tensor
from torchvision import transforms

IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']  # include image suffixes
def get_image(data_path):
    image_dict = {}
    # path_list = [data_path+'nonrumor_images/', data_path+'rumor_images/']
    path_list = [data_path + 'images/']
    for path in path_list:
        # data_transforms = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        for i, filename in enumerate(os.listdir(path)):
            try:
                im = path + filename
                # im = Image.open(path + filename).convert('RGB')
                # im = data_transforms(im)
                # image_dict[filename.split('/')[-1].split('.')[0]] = im  # remove '.jpg'
                image_dict[filename] = im  # remove '.jpg'
            except:
                print(filename)
    print("image length " + str(len(image_dict)))
    return image_dict


def count(labels):
    r, nr = 0, 0
    for label in labels:
        if label == 0:
            nr += 1
        elif label == 1:
            r += 1
    return r, nr


def get_data(data_path, mode, image_dict):
    # file = data_path+mode+'_data.csv'
    file = data_path + mode + '_data.xlsx'
    data = pd.read_excel(file)
    tweet = data['text'].tolist()
    image_url = data['images_list'].tolist()
    label = data['label'].tolist()

    texts, images, labels = [], [], []
    ct = 0
    for url in image_url:
        image = url.split('\t') if isinstance(url, str) else []
        for img in image:
            if img in image_dict:
                texts.append(tweet[ct].split())
                images.append(image_dict[img])
                labels.append(label[ct])
                break
        # image_dict_tensors = []
        # image_temp = torch.tensor(np.ones((3, 224, 224), dtype=np.uint8) * 255)
        # for img in image:
        #     if img in image_dict:  # 有照片
        #         img = image_dict[img]
        #         image_dict_tensors.append(img)
        # if len(image_dict_tensors) == 1:
        #     image_temp = image_dict_tensors[0]
        # elif len(image_dict_tensors) > 1:
        #     temp = torch.concat(image_dict_tensors, dim=1)
        #     resized_images = F.interpolate(temp.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        #     image_temp = resized_images.squeeze(0)
        #
        # texts.append(tweet[ct].split())
        # images.append(image_temp)
        # labels.append(label[ct])
        ct += 1
    print('weibo:', len(texts), 'samples...')

    r, nr = count(labels)
    print(mode, 'contains:', r, 'rumor tweets,', nr, 'real tweets.')

    rt_data = {'text': texts, 'image': images, 'label': labels}
    return rt_data


def get_vocab(train_data, test_data):
    vocab = defaultdict(float)
    all_text = train_data['text'] + test_data['text']
    for sentence in all_text:
        for word in sentence:
            vocab[word] += 1
    return vocab, all_text


"""refer to EANN"""


def add_unknown_words(w2v, vocab, min_df=1, k=32):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in w2v and vocab[word] >= min_df:
            w2v[word] = np.random.uniform(-0.25, 0.25, k)


def get_W(w2v, k=32):
    word_idx_map = dict()
    W = np.zeros(shape=(len(w2v) + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in w2v:
        # for word in w2v.key_to_index.keys():
        W[i] = w2v[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def word2vec(text, word_idx_map, seq_len):
    word_embedding = []
    for sentence in text:
        sentence_embed = []
        for i, word in enumerate(sentence):
            sentence_embed.append(word_idx_map[word])
        # padding
        while len(sentence_embed) <= seq_len:
            sentence_embed.append(0)
        # cutting
        sentence_embed = sentence_embed[:seq_len]
        word_embedding.append(copy.deepcopy(sentence_embed))
    return word_embedding


def load_data(args):
    # np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    print('Loading image...')
    image_dict = get_image(args.data_path)

    print('Loading data...')
    train_data = get_data(args.data_path, 'train', image_dict)
    # test_data = get_data(args.data_path, 'test', image_dict)
    test_data = {'text': [], 'image': [], 'label': []}

    vocab, all_text = get_vocab(train_data, test_data)
    print("vocab size: " + str(len(vocab)))
    max_len = len(max(all_text, key=len))
    print("max sentence length: " + str(max_len))

    print("Loading word2vec...")
    word_embedding_path = args.data_path + 'w2v.pickle'

    with open(word_embedding_path, 'rb') as f:
        w2v = pickle.load(f, encoding='latin1')

    print("Number of words already in word2vec:" + str(len(w2v)))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    args.vocab_size = len(vocab)

    print('Translate text to embedding...')
    train_data['text_embed'] = word2vec(train_data['text'], word_idx_map, args.seq_len)
    test_data['text_embed'] = word2vec(test_data['text'], word_idx_map, args.seq_len)

    text = all_text
    text_embed = train_data['text_embed'] + test_data['text_embed']
    image = train_data['image'] + test_data['image']
    label = train_data['label'] + test_data['label']

    return np.array(text_embed), np.array(image), np.array(label), W
