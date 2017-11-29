import numpy as np
import collections
import os, unicodedata, re, string
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def load_lable_data(path):
    x_train, y_train = [], []
    length = 0
    index = ''
    with open(path, 'r') as data:
        lines = data.read().splitlines()
        for line in lines:
            start = line.find('+++$+++')
            content = normalizeString(line[start+8:])
            y_train.append(line[0])
            x_train.append(content)
    return (x_train, y_train)

def load_nolable_data(path):
    nolabel_data = []
    with open(path, 'r') as data:
        lines = data.read().splitlines()
        for line in lines:
            content = normalizeString(line)
            nolabel_data.append(content)
    return nolabel_data

def load_test_data(path):
    test_data = []
    with open(path, 'r') as data:
        lines = data.read().splitlines()
        lines = lines[1:]
        for line in lines:
            start = line.find(',')
            content = line[start+1:]
            test_data.append(content)
    return test_data

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([!])", r"!", s)
    s = re.sub(r"([?])", r"?", s)
    s = re.sub(r"([.])", r".", s)
    s = re.sub(r"([,])", r",", s)
    s = re.sub(r"([~])", r"~", s)
    s = re.sub(r"[0-9]", r"0", s)
    s = re.sub(r"[^a-zA-Z.!?0'~]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def data_process(data_dir):
    lable_data_dir = os.path.join(data_dir, 'training_lable.txt')
    nolable_data_dir = os.path.join(data_dir, 'training_nolable.txt')
    test_data_dir = os.path.join(data_dir, 'testing_data.txt')

    (label_x,label_y) = load_lable_data(lable_data_dir)
    nolabel_data = load_nolable_data(nolable_data_dir)
    test_data = load_test_data(test_data_dir)

    #
    # words = set()
    # for i, sentence in enumerate(label_x):
    #     for j, word in enumerate(sentence):
    #         word = normalizeString(word)
    #         label_x[i][j] = word
    #         words.add(word)
    # for i, sentence in enumerate(nolabel_data):
    #     for j, word in enumerate(sentence):
    #         word = normalizeString(word)
    #         nolabel_data[i][j] = word
    #         words.add(word)
    # for i, sentence in enumerate(test_data):
    #     for j, word in enumerate(sentence):
    #         word = normalizeString(word)
    #         test_data[i][j] = word
    #         words.add(word)
    # words = list(words)
    label_sequence, nolabel_sequence, test_sequence, word_num, sequence_length = gen_sequence(label_x, nolabel_data, test_data)
    # label_sequence = np.reshape(np.array(label_sequence), (-1, sequence_length, 1))
    # nolabel_sequence = np.reshape(np.array(nolabel_sequence), (-1, sequence_length, 1))
    # test_sequence = np.reshape(np.array(test_sequence), (-1, sequence_length, 1))
    # label_y = np.reshape(np.array(label_y), (-1, 1, 1))
    return label_sequence, nolabel_sequence, test_sequence, word_num, sequence_length, label_y

def gen_sequence(label_x, nolabel_data, test_data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(label_x)
    word_num = len(tokenizer.word_index) + 1

    label_sequence = tokenizer.texts_to_sequences(label_x)
    nolabel_sequence = tokenizer.texts_to_sequences(nolabel_data)
    test_sequence = tokenizer.texts_to_sequences(test_data)

    label_sequence = pad_sequences(label_sequence)
    sequence_length = label_sequence.shape[1]
    nolabel_sequence = pad_sequences(nolabel_sequence, maxlen=sequence_length)
    test_sequence = pad_sequences(test_sequence, maxlen=sequence_length)

    return label_sequence, nolabel_sequence, test_sequence, word_num, sequence_length

if __name__ == '__main__':
    base_dir = './'
    data_dir = os.path.join(base_dir, 'data')
    label_x, label_y, nolabel_data, test_data = data_process(data_dir)


    # print(tokenizer.word_index)
