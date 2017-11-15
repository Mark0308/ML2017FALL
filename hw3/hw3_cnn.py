import numpy as np
import pandas as pd
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import regularizers
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import Dense, Activation, Conv2D,  Convolution2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils, plot_model
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

def read_train_data (path):
    data = pd.read_csv(path)
    train_data = data.as_matrix()
    y_train = train_data[:, 0]
    x_train = []
    for feature in train_data[:, 1]:
        x_train.append(feature.split())
    x_train = np.array(x_train)
    x_train = x_train.astype('float32')
    return (x_train, y_train) #shape (28709, 2304), (28709,)




if __name__ == '__main__':

    # read_data
    x_train, y_train = read_train_data(sys.argv[1])

    # normalize
    x_train = x_train / 255

    # preprocessing
    x_train = x_train.reshape(-1, 48, 48, 1)

    y_train = np_utils.to_categorical(y_train)
    x_vali = x_train[:2600]
    y_vali = y_train[:2600]
    x_train = x_train[2600:]
    y_train = y_train[2600:]
    img_data_generator = ImageDataGenerator(rotation_range=30,
                                            zoom_range=[0.7, 1.3],
                                            horizontal_flip=True,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1,
                                            shear_range=0.1)
    img_data_generator.fit(x_train)
    # create model
    model = Sequential()

    # first CNN
    model.add(Conv2D(64,(3, 3),activation='relu',input_shape=(48, 48, 1),padding='same') ) # (1, 48, 48) -> (32, 46, 46)
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2 ))  # (32, 46, 46) -> (32, 23, 23)
    model.add(Dropout(0.2))
    # second CNN
    model.add(Conv2D(128,(3, 3),activation='relu',padding='same') )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2 )) # (50, 21, 21) -> (50, 10, 10)
    model.add(Dropout(0.3))
    # Third CNN
    model.add(Conv2D(256,(3, 3),activation='relu',padding='same') )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2 )) # (50, 21, 21) -> (50, 10, 10)
    model.add(Dropout(0.35))
    # Forth CNN
    model.add(Conv2D(512,(3, 3),activation='relu',padding='same') )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2 ))
    model.add(Dropout(0.4))

    # add fully connected
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # add output layer
    model.add(Dense(7))
    model.add(Activation('softmax'))

    es = [ModelCheckpoint(filepath='best_model.h5', monitor='val_acc', save_best_only= True)]
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

    # print model detail
    model.summary()
    plot_model(model, to_file='model_structure.png')
    history = model.fit_generator(img_data_generator.flow(x_train, y_train, batch_size=128),
                                                     epochs=100,
                                                     verbose=1,
                                                     callbacks=es,
                                                     steps_per_epoch=5*x_train.shape[0]//128,
                                                     validation_data=(x_vali,y_vali))

    model.save ('model')
    # history graph
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('history.png')


    #
    # predict_result = model.predict(x_test, batch_size=100)
    # predict_result = np.argmax(predict_result, axis = 1)
    # text = 'id,label\n'
    # for i in range(predict_result.shape[0]):
    #     text = text + str(i) + ',' + str(predict_result[i]) + '\n'
    # with open('result.csv', 'w') as output:
    #     output.write(text)
