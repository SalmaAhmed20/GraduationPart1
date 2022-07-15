import os
import unittest
from collections import deque
import mediapipe as mp

import cv2

from Detectionphase import classification

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class MyTestCase(unittest.TestCase):

    def test_smoking(self):
        print("[INFO] Loading Smoking model ...")
        class_model = classification.Model()
        pathimage = "D:\\Graduation project\\Graduation Part1\\dataset\\testing_data\\abc005.jpg"
        image =cv2.imread(pathimage)
        self.assertEqual(True, class_model.image_classification(image))

    def test_Nonsmoking(self):
        print("[INFO] Loading Smoking model ...")
        class_model = classification.Model()
        pathimage = "D:\\Graduation project\\Graduation Part1\\dataset\\testing_data\\abc202.jpg"
        image = cv2.imread(pathimage)
        self.assertEqual(False, class_model.image_classification(image))# add assertion here

if __name__ == '__main__':
    unittest.main()
