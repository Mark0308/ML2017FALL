import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys

data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
	data.append([])

n_row = 0
text = open('train.csv', 'r', encoding='utf8') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))	
    n_row = n_row+1
text.close()

x = []
y = []
# 每 12 個月
for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        # 18種污染物
            # 連續9小時
        for s in range(9):
            x[471*i+j].append(data[9][480*i+j+s] )
        y.append(data[9][480*i+j+9])
x = np.array(x)
y = np.array(y)

# add square term
# x = np.concatenate((x,x**2), axis=1)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

w = np.zeros(len(x[0]))
l_rate = 10
repeat = 100000

w_expect = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x)),x.transpose()),y)
# print (w)

#Start Training
x_t = x.transpose()
s_gra = np.zeros(len(x[0]))
for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo - y
    cost = np.sum(loss**2) / len(x)
    cost_a  = math.sqrt(cost)
    gra = np.dot(x_t,loss)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada
    print ('iteration: %d | Cost: %f  ' % ( i,cost_a))

# hypo = np.dot(x[3000:5652],w)
# loss = hypo - y[3000:5652]
# cost = np.sum(loss**2) / len(x[3000:5652])
# cost_a = math.sqrt(cost)
# print ('Cost: %f ' % (cost_a))
# print (w_expect)
# print (w)
#Save and Read Model
# # save model
# np.save('model.npy',w)
# # read model
# w = np.load('model.npy')

#Testing
test_x = []
n_row = 0
text = open('test.csv' ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])
        # for i in range(2,11):
        #     test_x[n_row//18].append(float(r[i]) )
    elif (n_row - 9) % 18 == 0:
        for i in range (2,11):
            test_x[n_row//18].append(float(r[i]) )
    n_row = n_row+1
text.close()
test_x = np.array(test_x)

# add square term
# test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)


ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

filename = "predict.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()