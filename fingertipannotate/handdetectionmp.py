import cv2
import mediapipe as mp
from glob import glob
import os
import numpy as np
from google.protobuf.json_format import MessageToDict
import pandas as pd

dfObj = pd.DataFrame(columns=['Name', 'Point0', 'Point1', 'Score'])
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For static images:
images = glob(
    os.path.join("/home/nutfruit/Documents/Fathima599_Fall20/AllVideos/Minh-images/validation/NotDetected", '*.jpg'))

file_list = {name: 0 for name in images}

counter = 0
m1 = len(file_list)


with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(sorted(file_list)):
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        t = 0
        for hand_landmarks in results.multi_hand_landmarks:
            response = results.multi_handedness[t]
            t += 1
            tags = response.classification
            serializable_tags = [MessageToDict(tag) for tag in tags]
            if serializable_tags[0].get('label') != 'Right':
                continue
            if serializable_tags[0].get('score') < 0.8:
                continue
        l = 0
        t = 0
        points = np.zeros((2, 2), dtype=np.float32)
        points[:] = -5
        scores = np.zeros(2, dtype=np.float32)
        for hand_landmarks in results.multi_hand_landmarks:
            response = results.multi_handedness[t]

            tags = response.classification
            serializable_tags = [MessageToDict(tag) for tag in tags]
            if serializable_tags[0].get('label') != 'Right':
                continue
            if serializable_tags[0].get('score') < 0.8:
                continue
            scores[t] = serializable_tags[0].get('score')
            t += 1
            if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP] is not None:
                points[l][0] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
                points[l][1] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
                if points[l][0] < 0 or points[l][1] < 0:
                    points[l][0] = -5
                    points[l][1] = -5
                l += 1
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )
            mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if points[0][0] == -5 and points[0][1] == -5 and points[1][0] == -5 and points[1][1] == -5:
            continue
        new_points = np.zeros((2, 2), dtype=np.float32)
        score = 0
        if l > 1:
            if scores[0] > scores[1] and points[0][0] != -5 and points[0][1] != -5:
                new_points[0][0] = points[0][0]
                new_points[0][1] = points[0][1]
                score = scores[0]
            elif scores[1] > scores[0] and points[1][0] != -5 and points[1][1] != -5:
                new_points[0][0] = points[1][0]
                new_points[0][1] = points[1][1]
                score = scores[1]
            else:
                continue
        else:
            new_points[0][0] = points[0][0]
            new_points[0][1] = points[0][1]
            score = scores[0]
        dfObj = dfObj.append({'Name': file, 'Point0': new_points[0][0], 'Point1': new_points[0][1], 'Score': score}, ignore_index=True)
        counter += 1
        cv2.imshow('annotated_image', annotated_image)
        cv2.waitKey(20)        

dfObj.to_csv(r'vnd.csv', index=False, header=True)
print(counter)
print(m1)
