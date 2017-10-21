import pandas as pd
import numpy as np
import sys

def ReadData (Trainfile, Resultfile, Testfile):
    # Read Data
    x = pd.read_csv(Trainfile).as_matrix() #shape (32561, 106)
    y = pd.read_csv(Resultfile).as_matrix() #shape (32561, 1)
    test = pd.read_csv(Testfile).as_matrix() #shape (10281, 106)
    x = np.delete(x, range(53, 59), 1)
    test = np.delete(test, range(53, 59), 1)
    y = y.reshape(y.shape[0]) #shape (32561, )
    # # add square term
    # x = np.concatenate((x, x**2), axis=1)

    # add bias
    # x = np.concatenate((np.ones((x.shape[0], 1)),x), axis=1)

    # Normalization
    x_all = np.concatenate((x, test))
    mean = np.mean(x_all, axis = 0)
    std = np.std(x_all, axis = 0)
    x_all = (x_all - mean) / (std + 1e-100)
    x, test = x_all[0 : x.shape[0]], x_all[x.shape[0] : x_all.shape[0]]
    return (x, y, test)

def Calculate_loss (x, y, w, bias):
    fwb = Sigmoid(np.dot(x, w) + bias)
    return (y - fwb, fwb)

def Sigmoid (z):
    return np.clip((1 / (1.0 + np.exp(-z))), 0.00000000000001, 0.999999999999999)

def Train (x, y):
    # Declare elements
    learning_rate = 0.5
    repeat = 10000
    w = np.zeros(len(x[0])) # shape (Feature Counts, 1)
    s_gra = np.zeros(len(x[0])) # shape (Feature Counts, 1)
    b_gra = 0.0
    bias = 1
    for i in range(repeat):
        error, fwb = Calculate_loss(x, y, w, bias)

        gra = -(np.dot(x.T, error))
        bgra = -np.sum(error)

        s_gra += gra **2
        b_gra += bgra **2

        ada = np.sqrt(s_gra)
        bada = np.sqrt(b_gra)

        w -= learning_rate * gra/ada
        bias -= learning_rate * bgra/bada
        # cross_entropy = Cross_Entropy(fwb, y)
        print ("\riteration = %d" %(i), end='', flush= True)
    return (w, bias)

def Test (w, test):
    hypo = np.dot(w, test.T) + bias
    fwb = Sigmoid(hypo)
    return fwb

def Cross_Entropy (f, y):
    return -np.mean((y * np.log(f+1e-100)) + (1 - y) * np.log(1 - f+1e-100))

if __name__ == '__main__':
    x, y, test = ReadData('./Data/X_train.csv', './Data/Y_train.csv', './Data/X_test.csv')
    w, bias = Train(x, y)
    result = Test (w, test)
    text = 'id,label\n'

    #Initial text
    for i in range (result.shape[0]):
        if result[i] > 0.5:
            result[i] = 1
        else:
            result[i] = 0
        text = text + str(i+1) + ',' + str(int(result[i])) + '\n'

    #Write Data
    with open('./Result/predict_logistic.csv', 'w') as output:
        output.write(text)
