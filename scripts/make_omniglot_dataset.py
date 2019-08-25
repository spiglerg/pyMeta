import tensorflow as tf
import sonnet as snt

import numpy as np

import matplotlib.pyplot as plt

from os import listdir
import os.path
from os.path import isfile, join
import cv2

import pickle



def make_dataset(folder, resize=28): # resize=-1 to disable
    allX = []
    allY = []

    alphabets_folders = listdir(folder)
    for alphabet_index, alphabet in enumerate(alphabets_folders):
        X = []
        Y = []

        characters_folders = listdir(join(folder, alphabet))
        for char_id, char in enumerate(characters_folders):
            samples = listdir(join(folder, alphabet, char))
            for s in samples:
                if os.path.splitext(s)[1]=='.png':
                    image = cv2.imread(join(folder, alphabet, char, s), cv2.IMREAD_GRAYSCALE)
                    if resize > 0:
                        image = cv2.resize(image, (resize, resize))
                    X.append(image)
                    Y.append(char_id)
        X = np.asarray(X)
        Y = np.asarray(Y)

        X = np.expand_dims(X, axis=-1)

        allX.append(X)
        allY.append(Y)

    return allX, allY



basefolder = "datasets/omniglot"


trainX, trainY = make_dataset(os.path.join(basefolder,'images_background')) ## trainX[alphabet_number][elementx (characters and repetitions]
testX, testY = make_dataset(os.path.join(basefolder,'images_evaluation'))


print( len(trainX) )
print( trainX[0].shape )
print( len(trainY) )
print( len(testX) )
print( len(testY) )


## It is now possible to sample tasks (training or testing), and then sample minibatches within them.


with open(os.path.join(basefolder,'omniglot.pkl'), 'wb') as f:
    pickle.dump({'trainX':trainX,'trainY':trainY,'testX':testX,'testY':testY}, f, pickle.HIGHEST_PROTOCOL)

# pickle.load(f)




