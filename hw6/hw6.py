import numpy as np
import pandas as pd
np.random.seed(1337)  # for reproducibility
# from keras.models import Model
# from keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
# import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os, sys

# def autoencoder():
#
# 	encoder_dim = 2
#
# 	# this is our input placeholder
# 	input_img = Input(shape=(784,))
#
# 	# cnn = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
# 	# cnn = BatchNormalization()(cnn)
# 	# cnn = Dropout(0.5)(cnn)
#
# 	# encoder layers
#
# 	encoded = Dense(256, activation='relu')(input_img)
# 	encoded = Dense(128, activation='relu')(encoded)
# 	encoded = Dense(64, activation='relu')(encoded)
# 	encoded = Dense(32, activation='relu')(encoded)
# 	encoder_output = Dense(16, activation='relu')(encoded)
#
# 	# decoder layers
# 	decoded = Dense(32, activation='relu')(encoder_output)
# 	decoded = Dense(64, activation='relu')(decoded)
# 	decoded = Dense(128, activation='relu')(decoded)
# 	decoded = Dense(256, activation='relu')(decoded)
#
# 	decoded = Dense(784, activation='sigmoid')(decoded)
#
# 	# construct the autoencoder model
# 	autoencoder = Model(input=input_img, output=decoded)
#
# 	# construct the encoder model for plotting
# 	encoder = Model(input=input_img, output=encoder_output)
#
# 	# compile autoencoder
# 	autoencoder.compile(optimizer='adam', loss='mse')
# 	return autoencoder, encoder

if __name__ == '__main__':
	train_data = np.load(sys.argv[1]) / 255
	train_data = np.reshape(train_data, (len(train_data), -1))
	# visual_data = np.load('visualization.npy') / 255
	# x_train = train_data[:130000]
	# x_val = train_data[130000:]
	# autoencoder, encoder = autoencoder()
    #
	# training
	# autoencoder.fit(train_data, train_data,
	#                 epochs=50,
	#                 batch_size=256,
	#                 shuffle=True)

	# # feature
	# feature = encoder.predict(train_data)
	# feature = feature.reshape(feature.shape[0], -1)
	# print('Start TSNE')
	# feature = TSNE(n_jobs=7).fit_transform(train_data)
	# print('TSNE finish')

	pca = PCA(n_components=250, whiten=True, svd_solver='randomized')
	feature = pca.fit_transform(train_data)
	print(feature.shape)

	print('Start kmeans ...')
	# kmeans
	kmean = KMeans(n_clusters=2, random_state=0, max_iter=1000)
	kmeans = kmean.fit(feature)

	f = pd.read_csv(sys.argv[2])
	IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])

	o = open(sys.argv[3], 'w')
	o.write('ID,Ans\n')

	for idx, i1, i2 in zip(IDs, idx1, idx2):
		p1 = kmeans.labels_[i1]
		p2 = kmeans.labels_[i2]
		if p1 == p2:
			pred = 1
		else:
			pred = 0
		o.write('{},{}\n'.format(idx,pred))
	o.close()

	# # plotting
	# visual_feature = pca.transform(visual_data)
	# visual_label = kmean.predict(visual_feature)
    #
	# print(visual_label)
	# labels = np.ones(10000)
	# for i in range(5000):
	# 	labels[i] = 0
	# count = 0
	# for i in range(10000):
	# 	if visual_label[i] == labels[i]:
	# 		count += 1
    #
	# print(count)
	# plt.scatter(visual_feature[:, 0], visual_feature[:, 1], c=visual_label)
	# plt.show()
