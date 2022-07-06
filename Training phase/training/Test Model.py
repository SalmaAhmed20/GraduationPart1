import cv2
from keras.utils import image_dataset_from_directory

from jgc.classification import Model
import visualkeras

classfication =Model()
img1=cv2.imread("D:\\Graduation project\\Graduation Part1\\dataset\\testing_data\\abc003.jpg")
print(classfication.image_classification(img1))
img2=cv2.imread("D:\\Graduation project\\Graduation Part1\\dataset\\testing_data\\abc202.jpg")
print(classfication.image_classification(img2))
