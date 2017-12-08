import os, math, time, datetime, sys, pickle
from load_data import load_test_data
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import Dense, Dropout, Activation

if __name__ == '__main__':
    # data_dir = '../data'
    # model_dir = '../model'
    # result_dir = '../result'
    #
    # model_path = os.path.join(model_dir, 'final.hdf5')

    model = load_model('best.hdf5')
    test_data = load_test_data(sys.argv[1])
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    test_sequence = tokenizer.texts_to_sequences(test_data)
    test_sequence = pad_sequences(test_sequence, maxlen = 37)

    result = model.predict(test_sequence)

    filestr = 'id,label\n'
    for i, ans in enumerate(sys.argv[1]):
        if float(ans) > 0.5:
            ans = 1
        else:
            ans = 0
        filestr = filestr + str(i) + ',' + str(ans) + '\n'
    with open(sys.argv[2], 'w') as output:
        output.write(filestr)
