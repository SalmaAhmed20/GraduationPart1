import time
from multiprocessing import Process, Queue

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras

from Detectionphase import classification, smoker


def Smoking_part1(Frames_data, Processing_times, Processed_frames):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        try:
            tf.config.experimental.set_memory_growth(gpus[1], True)
        except RuntimeError:
            print("RuntimeError in tf.config.experimental.list_physical_devices('GPU')")
    print("[INFO] Loading Smoking Detection Model ...")
    class_model = classification.Model()
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)
    bg2 = cv2.createBackgroundSubtractorMOG2(history=42, varThreshold=16, detectShadows=False)
    kg = cv2.createBackgroundSubtractorKNN(history=42, dist2Threshold=64, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    while True:
        print("in smoking...")
        if Frames_data.qsize() > 0:
            with mp_pose.Pose(
                    enable_segmentation=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
            ) as pose:
                outer_ROI = []
                is_inside = False
                ROI_ttl = 0
                frame = 0
                queue = Queue(10)
                q_count = 0
                hand_action_ttl = []
                hand_mouth_flag = False
                hand_mouth_ttl = 0
                hand_select = ''
                smoking_range = 0
                Smoker = smoker.Smoker()
            # To get region of interest upper body (face ,hand ,shoulders)
            image_data = Frames_data.get()
            t1 = time.time()
            Processing_times.put(time.time())
            frame = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, dsize=(960, 480)).astype("float32")
            image_height, image_width, _ = frame.shape
            cut_image = frame.copy()
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("fframes", frame)
            cv2.waitKey(0)
            results = pose.process(frame)
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
            # Head position
            if nose_x < l_ear_x and nose_x < r_ear_x:
                head_direction = 0
            elif nose_x > l_ear_x and nose_x > r_ear_x:
                head_direction = 1
            else:
                head_direction = -1
            # Width of the region of interest from shoulder to shoulder
            R_SHOULDER_coord = [int(R_shoulder.x * image_width), int(R_shoulder.y * image_height)]
            L_SHOULDER_coord = [int(L_shoulder.x * image_width), int(L_shoulder.y * image_height)]
            ROI_PADDING = abs(R_SHOULDER_coord[0] - L_SHOULDER_coord[0])
            # if sholders is not in our region of interest, then we need to adjustments
            if not is_inside:
                outer_ROI = [
                    int(Nose.x * image_width) - ROI_PADDING,
                    int(Nose.y * image_height) - ROI_PADDING,
                    ROI_PADDING * 2,
                    ROI_PADDING * 2
                ]
                is_inside = True
                ROI_ttl = time.time()
                Processing_times.put(ROI_ttl)
            elif is_inside:
                if outer_ROI[0] > int(Nose.x * image_width) \
                        or outer_ROI[0] + outer_ROI[2] < int(Nose.x * image_width):
                    is_inside = False
                    ROI_ttl = time.time()
                    Processing_times.put(ROI_ttl)
            ROI_cut_image = cut_image[outer_ROI[1]:outer_ROI[1] + outer_ROI[3],
                            outer_ROI[0]:outer_ROI[0] + outer_ROI[2]]
            if not queue.full():
                queue.put(class_model.image_classification(ROI_cut_image))
            if queue.full():
                label = queue.get()
                if label:
                    q_count += 1
                else:
                    q_count = 0
            if q_count > 3:
                image = cv2.rectangle(
                    frame,
                    (int(outer_ROI[0]), int(outer_ROI[1])),
                    (int(outer_ROI[0] + outer_ROI[2]), int(outer_ROI[1] + outer_ROI[3])),
                    (0, 0, 255), 2)
                image = cv2.rectangle(
                    image,
                    (int(outer_ROI[0]), int(outer_ROI[1] + outer_ROI[3])),
                    (int(outer_ROI[0]) + 360, int(outer_ROI[1] + outer_ROI[3] + 40)),
                    (255, 255, 255), -1)
                image = cv2.putText(
                    image,
                    'SMOKING Classification',
                    (int(outer_ROI[0]), int(outer_ROI[1] + outer_ROI[3] + 30)), 0, 1,
                    (0, 0, 255),
                    2
                )
                Processed_frames.put(image)


def Violence_Part2(Frames_data, Predicated_data, Processing_times):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            print("RuntimeError in tf.config.experimental.list_physical_devices('GPU')")
    times = []
    SUM = 0
    is_first_detection = True
    print("[INFO] Loading Violence model ...")
    model = keras.models.load_model('D:\\Graduation project\\Graduation Part1\\ResNet50.hdf5')
    while True:
        if Frames_data.qsize() != 0:
            image_data = Frames_data.get()
            t1 = time.time()
            Processing_times.put(t1)
            Processing_times.put(time.time())
            frame = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (128, 128)).astype("float32")
            frame = frame.reshape(128, 128, 3) / 255
            t2 = time.time()
            SUM += (t2 - t1)
            if is_first_detection == True:
                SUM = 0
                is_first_detection = False
                print("prediction started")
            preds = model.predict(np.expand_dims(frame, axis=0))[0]
            i = (preds > 0.50)[0]
            label = i
            Predicated_data.put(label)
        if Frames_data.qsize() == 0 and SUM != 0:
            print("SUM :", SUM)


def postprocess_mp(Predicted_data, original_frames, Processed_frames, Processing_times, input_size,
                   score_threshold, iou_threshold, rectangle_colors, realtime):
    times = []
    while True:
        if Predicted_data.qsize() != 0:
            pred_bbox = Predicted_data.get()
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
        else:
            break


def detect_video_realtime_mp(video_path, output_path, input_size=416, show=False,
                             score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', realtime=False):
    if realtime:
        vid = cv2.VideoCapture(0)
    else:
        vid = cv2.VideoCapture(video_path)

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))  # output_path must be .mp4
    no_of_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    original_frames = Queue()
    Frames_data = Queue()
    Predicted_data = Queue()
    Processed_frames = Queue()
    Processing_times = Queue()
    Final_frames = Queue()

    p1 = Process(target=Violence_Part2, args=(Frames_data, Predicted_data, Processing_times))
    p2 = Process(target=postprocess_mp, args=(
        Predicted_data, original_frames, Processed_frames, Processing_times, input_size, score_threshold,
        iou_threshold, rectangle_colors, realtime))
    p3 = Process(target=Show_Image_mp, args=(Processed_frames, True, Final_frames))
    p1.start()
    p2.start()
    p3.start()
    started = False

    while True:
        ret, img = vid.read()
        if not ret:
            break

        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_frames.put(original_image)

        image_data = np.copy(original_image)
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        Frames_data.put(image_data)
        while started == False and Frames_data.qsize() > 0:
            if Processed_frames.qsize() == 0:
                time.sleep(0.1)
                continue
            else:
                started = True
                start_time = time.time()
                break

    while True:
        print("Orginal frames", original_frames.qsize())
        print("Frames_data", Frames_data.qsize())
        print("Predicted_data", Predicted_data.qsize())
        print("Processed_frames", Processed_frames.qsize())
        print("Final_frames", Final_frames.qsize())
        if original_frames.qsize() == 0 and Frames_data.qsize() == 0 and Predicted_data.qsize() == 0 and Processed_frames.qsize() == 0 and Processing_times.qsize() == 0 and Final_frames.qsize() == 0:
            p1.terminate()
            p2.terminate()
            p3.terminate()
            break
        elif Final_frames.qsize() > 0:
            image = Final_frames.get()
            if output_path != '': out.write(image)
    end_time = time.time()
    print("total_duration", end_time - start_time)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_video_realtime_mp("D:\\Graduation project\\Graduation Part1\\WhatsApp Video 2022-07-07 at 8.38.19 AM.mp4",
                             "D:\\Graduation project\\Graduation Part1\\out")
