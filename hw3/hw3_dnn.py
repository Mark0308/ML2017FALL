import numpy as np
import pandas as pd
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
from keras.preprocessing.image import ImageDataGenerator

def read_train_data ():
    data = pd.read_csv("../hw3_data/train.csv")
    train_data = data.as_matrix()
    y_train = train_data[:, 0]
    x_train = []
    for feature in train_data[:, 1]:
        x_train.append(feature.split())
    x_train = np.array(x_train)
    x_train = x_train.astype('float32')
    return (x_train, y_train) #shape (28709, 2304), (28709,)


def read_test_data ():
    data = pd.read_csv('../hw3_data/test.csv')
    test_data = data.as_matrix()
    x_test = []
    for feature in test_data[:, 1]:
        x_test.append(feature.split())
    x_test = np.array(x_test)
    x_test = x_test.astype('float32')
    return (x_test)

if __name__ == '__main__':

    # read_data
    x_train, y_train = read_train_data()
    x_test = read_test_data()

    # normalize
    x_train = x_train / 255
    x_test = x_test / 255

    # preprocessing
    x_train = x_train.reshape(-1, 1, 48, 48)
    x_test = x_test.reshape(-1, 1, 48, 48)


    y_train = np_utils.to_categorical(y_train)
    x_vali = x_train[-3000:]
    y_vali = y_train[-3000:]
    x_train = x_train[:-3000]
    y_train = y_train[:-3000]


    # create model
    model = Sequential()
    # add fully connected
    model.add(Flatten(batch_input_shape=(None, 1, 48, 48)))
    model.add(Dense(1200))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.45))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(500))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # add output layer
    model.add(Dense(7))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

    # print model detail
    model.summary()
    plot_model(model, to_file='model_structure.png')
    history = model.fit(x_train, y_train, batch_size=64,
                                                     epochs=30,
                                                     shuffle=True,
                                                     verbose=1,
                                                     validation_data=(x_vali, y_vali))

    # history graph
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('history_dnn.png')


    model.save ('dnn_model')
    #
    # predict_result = model.predict(x_test, batch_size=100)
    # predict_result = np.argmax(predict_result, axis = 1)
    # text = 'id,label\n'
    # for i in range(predict_result.shape[0]):
    #     text = text + str(i) + ',' + str(predict_result[i]) + '\n'
    # with open('result.csv', 'w') as output:
    #     output.write(text)
