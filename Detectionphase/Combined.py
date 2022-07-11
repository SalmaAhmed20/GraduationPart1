import os

from APIS import APIS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras

from Detectionphase import classification, smoker


def Violence_Part2(Frame, model, scores):
    image_data = Frame
    frame = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (128, 128)).astype("float32")
    frame = frame.reshape(128, 128, 3) / 255
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    scores.append(preds)
    results = np.array(scores).mean(axis=0)
    label = (results > 0.60)[0]
    if label:
        text_color = (0, 0, 255)
    else:
        text_color = (0, 255, 0)
    text = "Violence: {}".format(label)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    image_data = cv2.putText(image_data, text, (35, 50), FONT, 1.25, text_color, 3)
    return image_data, scores


def Smoking_part2(v_path):
    ViolanceScore = deque(maxlen=128)
    SmokingScore = deque(maxlen=128)
    firebaseApi = APIS()

    mp_pose = mp.solutions.pose
    # Load our 2 models
    print("[INFO] Loading Violence model ...")
    model = keras.models.load_model('D:\\Graduation project\\Graduation Part1\\ResNet50.hdf5')
    print("[INFO] Loading Smoking model ...")
    class_model = classification.Model()

    bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)
    kg = cv2.createBackgroundSubtractorKNN(history=42, dist2Threshold=64, detectShadows=False)
    video_path = v_path
    try:
        video = cv2.VideoCapture(int(video_path))

    except:
        video = cv2.VideoCapture(video_path)
    # To Save videos in the ending of detection Process
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter('basicvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

    with mp_pose.Pose(
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
    ) as pose:
        outer_ROI = []
        is_inside = False
        ROI_ttl = 0
        frame_count = 0
        frame_count2 = 0
        hand_action_ttl = []
        hand_mouth_flag = False
        hand_mouth_ttl = 0
        hand_select = ''
        smoking_range = 0
        Smoker = smoker.Smoker()

        while video.isOpened():
            success, ori_image = video.read()
            if not success:
                print("video.read fail.")
                break
            frame_rate = video.get(cv2.CAP_PROP_FPS)
            ori_image = cv2.rotate(ori_image, cv2.ROTATE_180)

            ori_image, ViolanceScore = Violence_Part2(ori_image, model, ViolanceScore)
            re=np.array(ViolanceScore).mean(axis=0)
            vorn=(re > 0.60)[0]
            if frame_count >= 90 and vorn:
                firebaseApi.FirebaseAPI(False, "0")
                frame_count=0
                ViolanceScore.clear()

            cut_image = ori_image.copy()
            image_height, image_width, _ = ori_image.shape
            gray = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            bg_mask = bg.apply(gray, 0, 0.00001)

            ori_image.flags.writeable = False
            image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
            results = pose.process(ori_image)
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

            nose_x = Nose.x * image_width
            l_ear_x = L_ear.x * image_width
            r_ear_x = R_ear.x * image_width
            head_direction = 0
            if nose_x < l_ear_x and nose_x < r_ear_x:
                head_direction = 0
            elif nose_x > l_ear_x and nose_x > r_ear_x:
                head_direction = 1
            else:
                head_direction = -1

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

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
                predss = class_model.image_classification(ROI_cut_image)
                SmokingScore.append(predss)
                if (predss):
                    cv2.putText(
                        ori_image,
                        'SMOKING Classification',
                        (30, 80), 0, 1,
                        (0, 0, 255),
                        2
                    )
                else:
                    cv2.putText(
                        ori_image,
                        'No Smoking',
                        (30, 80), 0, 1,
                        (0, 255, 0),
                        2
                    )
                re = np.array(SmokingScore).mean(axis=0)
                vorn = (re > 0.60)
                if frame_count2 >= 90 and vorn:
                    firebaseApi.FirebaseAPI(True, "0")
                    frame_count2 = 0
                    SmokingScore.clear()
            # cv2.imshow('Smoking Detection Project', ori_image)
            writer.write(ori_image)
            frame_count += 1
            frame_count2+=1
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cv2.destroyAllWindows()
        video.release()


def postprocess_mp(Predicted_data1, original_frames, Processed_frames, Processing_times, input_size,
                   score_threshold, iou_threshold, rectangle_colors, realtime):
    times = []
    while True:
        if Predicted_data1.qsize() > 0:
            pred_bbox = Predicted_data1.get()
            if realtime:
                while original_frames.qsize() > 1:
                    original_image = original_frames.get()
            else:
                original_image = original_frames.get()
            times.append(time.time() - Processing_times.get())
            times = times[-20:]

            ms = sum(times) / len(times) * 1000
            fps = 1000 / ms
            if pred_bbox:
                text_color = (0, 0, 255)
            else:
                text_color = (0, 255, 0)
            text = "Violence: {}".format(pred_bbox)
            FONT = cv2.FONT_HERSHEY_SIMPLEX
            image = cv2.putText(original_image, text, (35, 50), FONT, 1.25, text_color, 3)
            # cv2.imshow("DDD",image)
            # cv2.waitKey(0)
            Processed_frames.put(image)


def Show_Image_mp(Processed_frames, show, Final_frames):
    while True:
        if Processed_frames.qsize() > 0:
            image = Processed_frames.get()
            Final_frames.put(image)
            if show:
                cv2.imshow('output', image)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break


def detect_video_realtime_mp(video_path):
    Smoking_part2(video_path)


if __name__ == '__main__':
    detect_video_realtime_mp("D:\\Graduation project\\Graduation Part1\\VID20220709184458.mp4")
