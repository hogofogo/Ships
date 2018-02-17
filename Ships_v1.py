#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:01:36 2018

@author: vlad
"""

import os
os.chdir('/Users/vlad/Projects/ships-in-satellite-imagery')
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy import misc
import glob
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.models import load_model
# from kt_utils import *

'''
For source of data and description see
https://www.kaggle.com/rhammell/ships-in-satellite-imagery
* initial attempt to classify based on the given lables (partial ship images classfied as '0')
* I made the dataset more complicated by reclassifying partial ships as '1' label; note that ship images crossing the corner
  were typically classified as '0' unless it's very clear the image is a ship (typically needs to include a bow or a stern)
* poor initial results addressed by building more data for misclassified types of images
* increased size of model to address the increased image complexity
* 91%+ prediction on the test set and can be further improved
'''


np.random.seed(88)

#upload RGB images
image = []
for image_path in glob.glob("/Users/vlad/Projects/ships-in-satellite-imagery/shipsnet/*.png"):
    image.append(misc.imread(image_path))
X_orig = np.asarray(image)

#get list of file names
filename = []
filename.append(glob.glob("/Users/vlad/Projects/ships-in-satellite-imagery/shipsnet/*.png"))
filename = np.asarray(filename).T

#60th position in line corresponds to the label value => create labels list
y_orig = []
for n in range(len(filename)):
    line = np.array_str(filename[n])
    y_orig.append(line[60])
y_orig = np.asarray(y_orig).T
y_orig = y_orig.reshape((len(y_orig), 1))

#after label correcton get the following almost 1:1 split
#np.unique(y_orig, return_counts = True)    
#(array(['0', '1'], 
#       dtype='<U1'), array([1407, 1393]))

# Normalize image vectors
X_orig = X_orig/255.

# data in files are sequential: first 2100 are zeros, and then 700 are ones; shuffle x and y
X_orig, y_orig = shuffle(X_orig, y_orig, random_state=0)

#split into test and train sets
X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.2, random_state=1)


print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("y_train shape: " + str(y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("y_test shape: " + str(y_test.shape))

np.random.seed(88)

def ShipModel(input_shape):
    """
    Implementation of the model.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """   
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((2, 2))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(10, (3, 3), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    
     # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    
    # Padding -> CONV -> BN -> RELU Block applied to X -> MAXPOOL
    X = ZeroPadding2D((2, 2))(X)
    X = Conv2D(25, (3, 3), strides = (2, 2), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    
    # CONV -> BN -> RELU Block applied to X
    X = ZeroPadding2D((2, 2))(X)
    X = Conv2D(50, (3, 3), strides = (2, 2), name = 'conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)

   
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates Keras model instance
    model = Model(inputs = X_input, outputs = X, name='ShipModel')
    
    
    return model

#Create the model
shipModel = ShipModel(input_shape=(80, 80, 3))

#Compile the model to configure the learning process
shipModel.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#Train the model
shipModel.fit(x = X_train, y = y_train, epochs = 30, batch_size = 32)
shipModel.save('/Users/vlad/Projects/ships-in-satellite-imagery/shipModel.h5')
#shipModel = load_model('/Users/vlad/Projects/ships-in-satellite-imagery/shipModel.h5')


Test/evaluate the model
preds = shipModel.evaluate(x = X_test, y = y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

    
shipModel.summary()

plot_model(shipModel, to_file='ShipModel.png')
SVG(model_to_dot(shipModel).create(prog='dot', format='svg'))





#CREATE A DICTIONARY OF TYPES OF IMAGES
class_dict = {0:'Ship: no', 1:'Ship: yes'}

#CLASSIFY IMAGES BY SHIP OR NO SHIP
def classify_random_image():
    from PIL import Image
    import random
    import imageio
    image_path = filename[random.randrange(0,len(filename),1)][0]
    image = Image.open(image_path)
    image.show()
    my_image = imageio.imread(image_path)
    my_image = np.expand_dims(my_image, axis = 0)
    my_image = my_image/255.
    my_image_prediction = round(shipModel.predict(my_image)[0,0])
    print(class_dict[my_image_prediction])
    



#USEFUL UTILITIES TO CREATE MORE DATA IF NECESSARY
import re        
# for builk changing file names in a given directory
for f in os.listdir():
    new_f = '1' + re.sub(r'^0*', r'', f)
    os.rename(f, new_f)

from PIL import Image
# for rotation of selected images in a set directory, rotate 180 degrees
for f in os.listdir("/Users/vlad/Projects/rename backup"):
    if not f.startswith('.'):
        image_obj = Image.open("/Users/vlad/Projects/rename backup/" + f)
        rotated_image = image_obj.rotate(180)
        saved_location = re.sub(r'.png', r'_R180.png', "/Users/vlad/Projects/flip/" + f)
        rotated_image.save(saved_location)
   
