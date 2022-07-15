import cv2
import keras.saving.save
from fastai.layers import Flatten
from keras.layers import ZeroPadding2D, Dropout, BatchNormalization, Add, Rescaling
from keras.utils import image_dataset_from_directory

import visualkeras

from Detectionphase.classification import Model
from keras import layers
from collections import defaultdict
color_map = defaultdict(dict)
classfication =Model()
from keras.utils.vis_utils import plot_model

model = keras.models.load_model('D:\\Graduation project\\Graduation Part1\\ResNet50.hdf5')

plot_model(model, to_file='model2_plot.png', show_shapes=True, show_layer_names=True)

# img1=cv2.imread("D:\\Graduation project\\Graduation Part1\\dataset\\testing_data\\abc003.jpg")
# print(classfication.image_classification(img1))
# img2=cv2.imread("D:\\Graduation project\\Graduation Part1\\dataset\\testing_data\\abc202.jpg")
# print(classfication.image_classification(img2))
# color_map[layers.Dense]['fill'] = '#fb5607'
# visualkeras.layered_view(classfication.loaded_model,legend=True,to_file="models.png",type_ignore=[Rescaling],color_map=color_map).show("vo")
import cv2
from keras import Sequential
import os, os.path

from matplotlib import pyplot as plt
import tensorflow as tf
data_augmentation = Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip(
        'horizontal', input_shape=(180, 180, 3)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)
])
import os, os.path
DIR = "D:\\Graduation project\\Graduation Part1\\dataset\\training_data\\smoking"

# simple version for working with CWD
print (len([name for name in os.listdir(DIR) ]))
DIR = "D:\\Graduation project\\Graduation Part1\\dataset\\training_data\\notsmoking"

# simple version for working with CWD
print (len([name for name in os.listdir(DIR) ]))

# path joining version for other paths
# print (len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))
from keras.utils import image_dataset_from_directory

import os, os.path
dataset_path = "D:\\Graduation project\\Graduation Part1\\dataset\\training_data\\notsmoking"
print (len([name for name in os.listdir(dataset_path)]))
train_set = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset='training',
    seed=123,
    shuffle=True,

)
valid_set = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset='validation',
    seed=123,
    shuffle=True,

)
