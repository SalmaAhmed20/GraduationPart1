import cv2
import keras.saving.save
from fastai.layers import Flatten
from keras.layers import ZeroPadding2D, Dropout, BatchNormalization, Add
from keras.utils import image_dataset_from_directory

import visualkeras

from Detectionphase.classification import Model

classfication =Model()
img1=cv2.imread("D:\\Graduation project\\Graduation Part1\\dataset\\testing_data\\abc003.jpg")
print(classfication.image_classification(img1))
img2=cv2.imread("D:\\Graduation project\\Graduation Part1\\dataset\\testing_data\\abc202.jpg")
print(classfication.image_classification(img2))
visualkeras.layered_view(keras.models.load_model("D:\\Graduation project\\Graduation Part1\\ResNet50.hdf5"),legend=True,to_file="violence.png",type_ignore=[ZeroPadding2D,BatchNormalization, Dropout,Add]).show("vo")
