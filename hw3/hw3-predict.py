import numpy as np
import pandas as pd
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import regularizers
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils, plot_model
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

def read_test_data (path):
    data = pd.read_csv(path)
    test_data = data.as_matrix()
    x_test = []
    for feature in test_data[:, 1]:
        x_test.append(feature.split())
    x_test = np.array(x_test)
    x_test = x_test.astype('float32')
    return (x_test)

if __name__ == '__main__':

    # load_model
    # model1 = load_model('model')
    # model2 = load_model('model_1.h5')
    model3 = load_model('best_model.h5')
    # read_data
    x_test = read_test_data(sys.argv[1])

    # normalize
    x_test = x_test / 255

    x_test = x_test.reshape(-1, 48, 48, 1)
    print ('predicting')
    predict_result = model.predict(x_test, batch_size=100)
    # # preprocessing
    # x_test = x_test.reshape(-1, 1, 48, 48)
    #
    # print ('predicting')
    # # predict test result
    # predict_result1 = model1.predict(x_test, batch_size=100)
    # predict_result2 = model2.predict(x_test, batch_size=100)

    # predict_result = predict_result1 + predict_result2 + predict_result3
    predict_result = np.argmax(predict_result, axis = 1)
    text = 'id,label\n'
    # output result
    for i in range(predict_result.shape[0]):
        text = text + str(i) + ',' + str(predict_result[i]) + '\n'
    with open(sys.argv[2], 'w') as output:
        output.write(text)
