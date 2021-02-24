import os
import scipy.io as scio
from matplotlib.pyplot import imread
# from scipy.misc import imresize
import matplotlib.image as mpimg
import numpy as np
from keras.utils import to_categorical
import cv2
import matplotlib.pyplot as plt
from numpy import random

def one_hot(y_):
    return to_categorical(y_)

def get_pictures(filepath):
    path = os.listdir(filepath)
    path.remove('.DS_Store')
    pictures = []
    actor_name = []
    count = 0
    for actor in path:
        actor_path = filepath + actor + "/"
        actor_path = os.listdir(actor_path)
        for i in actor_path:
            pic = filepath + actor + "/" + i
            p = mpimg.imread(pic)
            pictures.append(p)
            actor_name.append(count)
        count += 1
    return np.array(pictures), one_hot(np.array(actor_name))

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    # r, g, b = rgb[0], rgb[1], rgb[2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray/255.

def resize_pictures(train_X):
    for i in range(len(train_X)):
        '''
        resize all pictures
        '''
        train_X[i] = cv2.resize(train_X[i], dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
        # train_X[i] = imresize(train_X[i], (200, 200))
        if len(train_X[i].shape) == 2:
            continue
        '''
        transfer all pictures into gray
        '''
        train_X[i] = rgb2gray(train_X[i])
    return train_X

def split_and_preprocess(train_X, train_Y):
    total_num = len(train_X)
    validation_set = set()
    test_set = set()
    print(total_num)


if __name__ == "__main__":

	filepath = 'actors/faces/'
	train_x, train_y = get_pictures(filepath)
	train_x = resize_pictures(train_x)
	split_and_preprocess(train_x, train_y)

    # filepath1 = 'actors/faces/'
    # path1 = os.listdir(filepath1)
    # path1.remove('.DS_Store')
    # pic1 = filepath1 + path1[3] + "/" + 'Alec_Baldwin_3210_1863.jpeg'
    # face = np.array(mpimg.imread(pic1))
    
    # filepath2 = 'actors/images/'
    # path2 = os.listdir(filepath1)
    # path2.remove('.DS_Store')
    # pic2 = filepath2 + path2[3] + "/" + 'Alec_Baldwin_3210.jpeg'
    # image = np.array(mpimg.imread(pic2))
    # cropped_face = image[1116:2874,1111:2869]
    # imgplot = plt.imshow(cropped_face)
    # plt.show()