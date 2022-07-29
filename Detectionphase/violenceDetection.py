from collections import deque

import keras
import  cv2
import numpy as np

scores = deque(maxlen=25)

print("[INFO] Loading Violence model ...")
model = keras.models.load_model('D:\\Graduation project\\Graduation Part1\\ResNet50.hdf5')
video = cv2.VideoCapture("D:\\Graduation project\\V_1.mp4")
# To Save videos in the ending of detection Process
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter('basicvideo4.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))
while video.isOpened():
    success, Frame = video.read()
    if not success:
        print("video.read fail.")
        break
    image_data = Frame
    frame = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (128, 128)).astype("float32")
    frame = frame.reshape(128, 128, 3) / 255
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    scores.append(preds)
    results = np.array(scores).mean(axis=0)
    print(results)
    label = (results > 0.5)[0]
    if label:
        text_color = (0, 0, 255)
    else:
        text_color = (0, 255, 0)
    text = "Violence: {}".format(label)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    Frame = cv2.putText(Frame, text, (35, 50), FONT, 2, text_color, 3)
    writer.write(Frame)


