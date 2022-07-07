# Google MediaPipe
# Pose tutorial
# body landmarks
# 0 : nose
# 2, 5 : left_eye, right_eye
# 7, 8: left_ear, right_ear
# 9 : mouth_left
# 10: mouth_right
# right hand 16: right_wrist 18:right_pinky 20: right_index 22: right_thumb
# left hand 15: left_wrist 17: left_pinky 19: left_index 21: left_thumb
# 11: left_shoulder 12: right_shoulder

# imports
import time
from collections import deque
from queue import Queue

import cv2
import mediapipe as mp
import numpy as np

from Detectionphase import classification, smoker

# mediapipe initialization
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
# for masking
bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)
bg2 = cv2.createBackgroundSubtractorMOG2(history=42, varThreshold=16, detectShadows=False)
kg = cv2.createBackgroundSubtractorKNN(history=42, dist2Threshold=64, detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

video_path = "D:/Graduation project/Graduation Part1/Smoking My First Cigarette in 3 Days.mp4"
video_name = video_path.split('/')[-1].split('.')[0]
print(video_name)
try:
    video = cv2.VideoCapture(int(video_path))
except:
    video = cv2.VideoCapture(video_path)

with mp_pose.Pose(
        enable_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
) as pose:
    outer_ROI = []
    is_inside = False
    ROI_ttl = 0
    frame = 0
    queue = Queue(3)
    dq = deque()
    hand_action_ttl = []
    hand_mouth_flag = False  # it is true if hand closer to mouth
    hand_mouth_ttl = 0
    hand_select = ''
    frame_rate = 0
    smoking_range = 0
    Smoker = smoker.Smoker()
    smoker_dict = Smoker.smoker_dictionary
    Smoking = 0
    class_model = classification.Model()
    while video.isOpened():
        success, ori_image = video.read()
        if not success:
            print("video.read fail.")
            break
        try:
            frame_rate = video.get(cv2.CAP_PROP_FPS)
        except:
            frame_rate = 24
        image = ori_image.copy()
        image = cv2.resize(image, dsize=(960, 480))
        cut_image = image.copy()
        image_height, image_width, _ = image.shape
        gray = cv2.resize(ori_image, dsize=(960, 480))
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        bg_mask = bg.apply(gray, 0, 0.00001)

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if not results.pose_landmarks:
            continue

        Nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        R_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
        L_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
        R_mouth = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT]
        L_mouth = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]
        R_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        L_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        L_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
        R_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
        L_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
        R_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]

        nose_x = Nose.x * image_width
        l_ear_x = L_ear.x * image_width
        r_ear_x = R_ear.x * image_width
        head_direction = 0
        if nose_x < l_ear_x and nose_x < r_ear_x:  # look right
            head_direction = 0
        elif nose_x > l_ear_x and nose_x > r_ear_x:  # look left
            head_direction = 1
        else:  # Forward
            head_direction = -1

        cv2.drawMarker(
            image,
            (int(R_hand.x * image_width), int(R_hand.y * image_height)),
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=42)
        cv2.drawMarker(
            image,
            (int(R_mouth.x * image_width), int(R_mouth.y * image_height)),
            (0, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=42)

        R_SHOULDER_coord = [int(R_shoulder.x * image_width), int(R_shoulder.y * image_height)]
        L_SHOULDER_coord = [int(L_shoulder.x * image_width), int(L_shoulder.y * image_height)]
        ROI_PADDING = abs(R_SHOULDER_coord[0] - L_SHOULDER_coord[0])  # // 3
        # ROI
        if not is_inside:
            outer_ROI = [
                int(Nose.x * image_width) - ROI_PADDING,
                int(Nose.y * image_height) - ROI_PADDING,
                ROI_PADDING * 2,
                ROI_PADDING * 2
            ]
            is_inside = True
            ROI_ttl = time.time()
        elif is_inside:
            if outer_ROI[0] > int(Nose.x * image_width) \
                    or outer_ROI[0] + outer_ROI[2] < int(Nose.x * image_width):
                is_inside = False
                ROI_ttl = time.time()

        cv2.rectangle(
            image,
            (outer_ROI[0], outer_ROI[1]),
            (outer_ROI[0] + outer_ROI[2], outer_ROI[1] + outer_ROI[3]),
            (255, 0, 0),
            2)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        r_hand_coord = [int(R_hand.x * image_width), int(R_hand.y * image_height)]
        l_hand_coord = [int(L_hand.x * image_width), int(L_hand.y * image_height)]
        r_mouth_coord = [int(R_mouth.x * image_width), int(R_mouth.y * image_height)]
        l_mouth_coord = [int(L_mouth.x * image_width), int(L_mouth.y * image_height)]
        if not hand_mouth_flag:
            if abs(r_hand_coord[1] - r_mouth_coord[1]) < abs(R_SHOULDER_coord[1] - r_mouth_coord[1]):
                hand_mouth_ttl = time.time()
                hand_mouth_flag = True
                hand_select = 'r'
            elif abs(l_hand_coord[1] - l_mouth_coord[1]) < abs(L_SHOULDER_coord[1] - l_mouth_coord[1]):
                hand_mouth_ttl = time.time()
                hand_mouth_flag = True
                hand_select = 'l'
        else:
            if hand_select == 'r' and abs(r_hand_coord[1] - r_mouth_coord[1]) > abs(
                    R_SHOULDER_coord[1] - r_mouth_coord[1]):
                hand_action_ttl.append(hand_mouth_ttl - time.time())
                hand_mouth_ttl = time.time()
                hand_mouth_flag = False
                if Smoker.if_in_dict(1):
                    Smoker.smoker_dictionary[1].smoking_point += 2
                    print("????" + str(Smoker.smoker_dictionary[1].smoking_point))
            elif hand_select == 'l' and abs(l_hand_coord[1] - l_mouth_coord[1]) > abs(
                    L_SHOULDER_coord[1] - l_mouth_coord[1]):
                hand_action_ttl.append(hand_mouth_ttl - time.time())
                hand_mouth_ttl = time.time()
                hand_mouth_flag = False
                if Smoker.if_in_dict(1):
                    Smoker.smoker_dictionary[1].smoking_point += 2
                    print("????" + str(Smoker.smoker_dictionary[1].smoking_point))
            smoking_range = frame

        th_image = []
        if (time.time() - ROI_ttl) > 4:
            bg2_mask = kg.apply(gray, 0, 0.025)
            sub_mask = cv2.bitwise_and(bg_mask, bg2_mask)

            if not Smoker.if_in_dict(1):
                Smoking = smoker.Smoking()
                Smoking.set_data([1, outer_ROI, bg2_mask, frame_rate])
                Smoker.add_dict(1, Smoking)

            crop_image = image[outer_ROI[1]:outer_ROI[1] + outer_ROI[3], outer_ROI[0]:outer_ROI[0] + outer_ROI[2]]
            ROI_cut_image = cut_image[outer_ROI[1]:outer_ROI[1] + outer_ROI[3],
                            outer_ROI[0]:outer_ROI[0] + outer_ROI[2]]
            # crop_image_binary == crop sub_mask(bg - bg2)
            crop_image_binary = sub_mask[outer_ROI[1]:outer_ROI[1] + outer_ROI[3],
                                outer_ROI[0]:outer_ROI[0] + outer_ROI[2]]
            cv2.imshow('crop_image_binary', crop_image_binary)

            ret, th_image = cv2.threshold(crop_image_binary, thresh=250, maxval=255, type=cv2.THRESH_BINARY)
            th_image = cv2.medianBlur(th_image, ksize=3)
            # cv2.imshow('th_image', th_image)

            if len(hand_action_ttl) > 1 and abs(hand_action_ttl[-1] - hand_action_ttl[-2]) < 5:
                # smoke detector
                conv_image = cv2.resize(th_image, dsize=(100, 100))
                retn, conv_image = cv2.threshold(conv_image, thresh=125, maxval=255, type=cv2.THRESH_BINARY)

                smoke_map = []
                smoke_map_append = smoke_map.append
                for i in range(0, 95, 5):
                    line = []
                    line_append = line.append
                    for j in range(0, 95, 5):
                        count = 0
                        for k in range(5):
                            for l in range(5):
                                if conv_image[i + k][j + l] > 125:
                                    count += 1
                        if count > 12:
                            line_append(255)
                        else:
                            line_append(0)
                    smoke_map_append(line)
                np_smoke_map_image = np.array(smoke_map).astype(np.uint8)
                resize_smoke_map = cv2.resize(np_smoke_map_image, dsize=(outer_ROI[2], outer_ROI[3]))
                cv2.imshow('smoke_map', resize_smoke_map)

                square_len = (outer_ROI[2] // 2) // 4

                if nose_x < r_ear_x and nose_x < l_ear_x:
                    square_x = (Nose.x * image_width) - outer_ROI[0]
                    square_y = (Nose.y * image_height) - outer_ROI[1]
                elif nose_x > r_ear_x and nose_x > l_ear_x:
                    square_x = (Nose.x * image_width) - outer_ROI[0] - square_len * 2
                    square_y = (Nose.y * image_height) - outer_ROI[1] - square_len * 2
                else:
                    square_x = (Nose.x * image_width) - outer_ROI[0] - square_len
                    square_y = (Nose.y * image_height) - outer_ROI[1] - square_len
                head_map_mask = np.zeros((outer_ROI[2], outer_ROI[3]), dtype=np.uint8)
                head_mask_box = np.array(
                    [[square_x, square_y - square_len // 2],
                     [square_x + square_len * 2, square_y - square_len // 2],
                     [square_x + square_len * 2, square_y + square_len * 2],
                     [square_x, square_y + square_len * 2]], dtype=np.int32
                )
                cv2.fillPoly(head_map_mask, [head_mask_box], color=(255, 255, 255))
                cv2.imshow('head_mask_map', head_map_mask)

                contours, _ = cv2.findContours(resize_smoke_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                mouth_padding = 0
                if head_direction == 0:
                    mouth_padding = -10
                elif head_direction == 1:
                    mouth_padding = 10
                Nose_coord = [Nose.x * image_width, Nose.y * image_height]
                R_mouth_coord = [R_mouth.x * image_width, R_mouth.y * image_height]
                L_mouth_coord = [L_mouth.x * image_width, L_mouth.y * image_height]
                find_smoke_contour = []
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if x < Nose_coord[0] + mouth_padding - outer_ROI[0] < x + w and \
                            y < Nose_coord[1] - outer_ROI[1] < y + h:
                        find_smoke_contour.append(cnt)
                    if x < R_mouth_coord[0] + mouth_padding - outer_ROI[0] < x + w and \
                            y < R_mouth_coord[1] - outer_ROI[1] < y + h:
                        find_smoke_contour.append(cnt)
                    if x < L_mouth_coord[0] + mouth_padding - outer_ROI[0] < x + w and \
                            y < L_mouth_coord[1] - outer_ROI[1] < y + h:
                        find_smoke_contour.append(cnt)

                cv2.drawContours(crop_image, contours, -1, (0, 255, 0), 2)
                if Smoker.smoker_dictionary[1].is_smoke(frame - smoking_range) and not hand_mouth_flag:
                    color = (255, 0, 0)
                    for cnt in find_smoke_contour:
                        mt = cv2.moments(cnt)
                        cx = int(mt['m10'] / mt['m00'])
                        cy = int(mt['m01'] / mt['m00'])
                        cv2.drawMarker(crop_image, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=42)
                        if cx < square_x or cx > (square_x + square_len * 2) or \
                                cy < square_y - square_len // 2 or cy > (square_y + square_len * 2):  # <<-- fix need
                            if L_SHOULDER_coord[1] > cy:
                                if not Smoker.smoker_dictionary[1].smoking_flag:
                                    Smoker.smoker_dictionary[1].smoking_point += 5
                                    # Smoker.smoker_dictionary[1].smoking_count += 1
                                    Smoker.smoker_dictionary[1].smoking_flag = True
                                color = (0, 0, 255)
                    cv2.drawContours(crop_image, find_smoke_contour, -1, color, 2)
                    print(Smoker.smoker_dictionary[1].smoking_point)
                    print(Smoker.smoker_dictionary[1].ROI_message)
                    Smoker.smoker_dictionary[1].is_smoking()
                    cv2.putText(
                        image,
                        Smoker.smoker_dictionary[1].ROI_message,
                        (int(outer_ROI[0]), int(outer_ROI[1] - 10)), 0, 0.75,
                        color,
                        2)
                else:
                    if Smoker.if_in_dict(1):
                        if (frame - smoking_range) > (frame_rate):
                            Smoker.smoker_dictionary[1].smoking_point = 0
                            Smoker.smoker_dictionary[1].smoking_flag = False
            else:
                pass
            if class_model.image_classification(ROI_cut_image):
                cv2.imshow('SMOKING', ROI_cut_image)
            cv2.imshow('crop_image', crop_image)
        else:
            if Smoker.if_in_dict(1):
                Smoker.del_dict(1)

        cv2.imshow('MediaPipe Pose', image)
        cv2.waitKey(0)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        frame += 1
    cv2.destroyAllWindows()
    video.release()
