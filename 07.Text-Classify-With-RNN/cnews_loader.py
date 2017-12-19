# -*- coding: utf-8 -*-

from collections import Counter
import tensorflow.contrib.keras as kr
import numpy as np


def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                contents.append(list(content))
                labels.append(label)
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    words = open_file(vocab_dir).read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id


def read_category():
    categories = ['体育', '财经', '房产', '家居',
                  '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id)
    return x_pad, y_pad


def get_content_ids(content, word_to_id, max_length=600):
    data_id = []
    data_id.append([word_to_id[x] for x in content if x in word_to_id])
    return kr.preprocessing.sequence.pad_sequences(data_id, max_length)


def get_batches(x, y, batch_size=64):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
