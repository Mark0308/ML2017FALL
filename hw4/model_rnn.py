import os, math, time, datetime, sys
from load_data import load_label_data
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation
import numpy as np


def LSTM_model(WORD_NUM, EMBEDDING_DIM, SEQUENCE_LENGTH):
    model = Sequential()
    model.add(Embedding(WORD_NUM, EMBEDDING_DIM, input_length=SEQUENCE_LENGTH))
    model.add(Bidirectional(LSTM(512, batch_input_shape=(None, SEQUENCE_LENGTH, EMBEDDING_DIM), return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(256)))
    model.add(Dropout(0.45))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    model_dir = './model'
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    best_path = os.path.join(model_dir, 'best.hdf5')
    model_path = os.path.join(model_dir, 'final.hdf5')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    x_train, label_y = load_label_data(sys.argv[1])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train)
    word_num = len(tokenizer.word_index) + 1

    label_sequence = tokenizer.texts_to_sequences(x_train)
    label_sequence = pad_sequences(label_sequence)
    sequence_length = label_sequence.shape[1]
    # label_sequence, nolabel_sequence, test_sequence, word_num, sequence_length, label_y = data_process(data_dir)
    model = LSTM_model(word_num, 100, sequence_length)
    with open('model_summary_rnn', 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    earlystopping = EarlyStopping(monitor='val_loss', patience = 10)
    checkpoint = ModelCheckpoint(filepath=best_path, save_best_only=True, monitor='val_loss')
    model.fit(label_sequence, label_y, batch_size=512, epochs=1,validation_split = 0.1)
    model.save(model_path)
    # model = load_model(model_path)

    # result = model.predict(test_sequence)
    #
    # filestr = 'id,label\n'
    # for i, ans in enumerate(result):
    #     if float(ans) > 0.5:
    #         ans = 1
    #     else:
    #         ans = 0
    #     filestr = filestr + str(i) + ',' + str(ans) + '\n'
    # with open('result.csv', 'w') as output:
    #     output.write(filestr)
