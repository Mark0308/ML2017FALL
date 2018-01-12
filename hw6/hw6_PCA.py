from skimage import transform
from skimage import io
import numpy as np
import os
import sys
import re

img_path = './img'
eigenface_path = './img/eigenfaces/'
face_path = './img/face'
draw = [1, 10, 100, 400]

def load_img(path):
    matrix = []

    for i in range(415):
        img = io.imread(path + str(i) + '.jpg')
        # img = transform.resize(img, (300, 300))
        img = img.flatten()
        matrix.append(img)
    matrix = np.array(matrix)
    return matrix

def draw_img(picture, path):
    picture = picture.reshape((600, 600, 3))
    picture -= np.min(picture)
    picture /= np.max(picture)
    picture = (picture * 255).astype(np.uint8)
    io.imsave(path, picture)

if __name__ == '__main__':

    # if not os.path.exists(img_path):
    #     os.makedirs(img_path)

    X_ori = load_img(sys.argv[1])

    X_mean = np.mean(X_ori, axis=1)
    X = X_ori.T
    U, s, V = np.linalg.svd(X - X_mean, full_matrices=False)

    # all_mean = np.mean(X_ori, axis=0)
    # all_mean = all_mean.reshape((600, 600, 3))
    # io.imsave('./img/all_mean.jpg', all_mean)

    eigenvalue_sum = np.sum(s)
    print(s[:4] / eigenvalue_sum)

    # for i in range(4):
    #     draw_img(U.T[i], eigenface_path + str(i) + '.jpg')

    # for i in draw:
    #
    #     weight = np.dot(X_ori[i] - X_mean[i], U[:,:4])
    #     img = np.matmul(U[:,:4], weight)
    #     img = img + X_mean[i]
    #     draw_img(img, face_path + str(i) + '.jpg')

    target_img = io.imread(os.path.join(sys.argv[1], sys.argv[2]))
    # target_img = transform.resize(target_img, (300, 300))
    target_img = target_img.flatten()
    target_mean = np.mean(target_img)
    target_index = int(re.search(r'\d+', sys.argv[2]).group())

    weight = np.dot(target_img - target_mean, U[:,:4])
    img = np.matmul(U[:,:4], weight)
    img = img + target_mean
    draw_img(img, 'reconstruction.jpg')
