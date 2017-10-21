import pandas as pd
import numpy as np
import sys

def ReadData (Trainfile, Resultfile, Testfile):
    # Read Data
    x = pd.read_csv(Trainfile).as_matrix() #shape (32561,106)
    y = pd.read_csv(Resultfile).as_matrix() #shape (32561, 1)
    test = pd.read_csv(Testfile).as_matrix()

    y = y.reshape(y.shape[0]) #shape (32561, )
    return (x, y, test)

def Calculate_Mu_Sigma (x, y):
    # Pick Data by y
    y_0, y_1 = (y == 0), (y == 1)
    x_0, x_1 = x[y_0, :], x[y_1, :] #shape (24720, 106) (7841, 106)
    count_0, count_1 = x_0.shape[0], x_1.shape[0]

    # Caculate mu
    mu_0 = x_0.sum(axis = 0) / count_0
    mu_1 = x_1.sum(axis = 0) / count_1

    # Caculate sigma
    # sigma_0 = np.matmul((x_0 - mu_0).T, (x_0 - mu_0)) / count_0
    # sigma_1 = np.matmul((x_1 - mu_1).T, (x_1 - mu_1)) / count_1
    sigma_0 = sigmaC(x_0, mu_0) / count_0
    sigma_1 = sigmaC(x_1, mu_1) / count_1
    sigma = (count_0/(count_0 + count_1) * sigma_0) + (count_1/(count_0 + count_1) * sigma_1)

    return (mu_0, mu_1, sigma, count_0, count_1)

def sigmaC (x, mu):
    sigma = np.zeros(shape = (106, 106))
    for i in x:
        sigma += np.dot(np.transpose([i - mu]), ([i - mu]))
    return sigma

def Caculate_z (mu_0, mu_1, sigma, test, count_0, count_1):
    sigma_inv = np.linalg.inv(sigma)
    w_T = np.dot(np.dot((mu_1 - mu_0).T, sigma_inv), test.T)
    b = (-0.5) * np.dot(np.dot(mu_1.T, sigma_inv), mu_1) + (0.5) * np.dot(np.dot(mu_0.T, sigma_inv), mu_0) + np.log(count_1/ count_0)
    z = w_T + b
    return z

if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print ('No data route')
    x, y, test = ReadData('./Data/X_train.csv', './Data/Y_train.csv', './Data/X_test.csv')
    mu_0, mu_1, sigma, count_0, count_1 = Calculate_Mu_Sigma(x, y) #Caculate mu, sigma
    z = Caculate_z(mu_0, mu_1, sigma, test, count_0, count_1) #Caculate z
    result = 1 / (1 + np.exp(-z)) #sigmoid function

    text = 'id,label\n' #Initial text
    for i in range (result.shape[0]):
        if result[i] > 0.5:
            result[i] = 1
        else:
            result[i] = 0
        text = text + str(i+1) + ',' + str(int(result[i])) + '\n'

    #Write Data
    with open('./Result/predict_generative.csv', 'w') as output:
        output.write(text)
