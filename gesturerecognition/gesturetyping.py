'''
TO RUN:
$env:FLASK_APP = "server.py"
flask run
You can modify the parameters, return values and data structures used in every function if it conflicts with your
coding style or you want to accelerate your code.
You can also import packages you want.
But please do not change the basic structure of this file including the function names. It is not recommended to merge
functions, otherwise it will be hard for TAs to grade your code. However, you can add helper function if necessary.
'''

from flask import Flask, request
from flask import render_template
import time
import json
import numpy as np
from scipy.interpolate import interp1d
import math
from operator import itemgetter
import cv2
import mediapipe as mp

from queue import PriorityQueue


def printKclosest(arr, n, x, k):
    # Make a max heap of difference with
    # first k elements.
    pq = PriorityQueue()
    for i in range(k):
        pq.put((-abs(arr[i] - x), i))

    # Now process remaining elements
    for i in range(k, n):
        diff = abs(arr[i] - x)
        p, pi = pq.get()
        curr = -p

        # If difference with current
        # element is more than root,
        # then put it back.
        if diff > curr:
            pq.put((-curr, pi))
            continue
        else:

            # Else remove root and insert
            pq.put((-diff, i))

    # Print contents of heap.
    result = []
    while not pq.empty():
        p, q = pq.get()
        result.append(q)
    return result


app = Flask(__name__)

# Centroids of 26 keys
centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240,
               170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50,
               120]
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# Pre-process the dictionary and get templates of 10000 words
words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])


protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
              [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
fingertip = []
threshold = 0.2
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


def get_keyboard():
    cap1 = cv2.VideoCapture(0)
    while cap1.isOpened():
        ret0, frame0 = cap1.read()
        if ret0:
            frame = np.copy(frame0)
            # converting BGR to GRAYSCALE
            dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

            # Initialize the detector parameters using default values
            parameters = cv2.aruco.DetectorParameters_create()

            # Detect the markers in the image
            markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary,
                                                                                  parameters=parameters)
            centroids = []
            width = 0
            for markers in markerCorners:
                for points in markers:
                    x = [p[0] for p in points]
                    y = [p[1] for p in points]
                    centroid = ((int)(sum(x) / len(points)), (int)(sum(y) / len(points)))
                    centroids.append(centroid)
                    width = max(y) - min(y)
                    cv2.circle(frame, centroid, radius=1, color=(0, 0, 255), thickness=-1)
            cv2.imshow("frame", frame)
            cv2.waitKey(100)
            if len(centroids) == 3 or len(centroids) == 4:
                cap1.release()
                cv2.destroyAllWindows()
                return centroids, frame0, width

    return markerCorners


# Centroids of 26 keys

charset1 = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P']
charset2 = ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L']
charset3 = ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
# header_list = ["x", "y"]
# df=pd.read_csv('alphabet_loc.csv', sep=',', names=header_list, header=None)
# np_arr = df[['x', 'y']].to_numpy()
centroids_X = np.zeros(26)
centroids_Y = np.zeros(26)
for i in range(0, 26):
    centroids_X[i] = np_arr[i][0]
    centroids_Y[i] = np_arr[i][1]
cords, image, width_marker = get_keyboard()
cords_flatten = np.array(cords).flatten()
min_x = min(cords_flatten[0], cords_flatten[2], cords_flatten[4])
max_x = max(cords_flatten[0], cords_flatten[2], cords_flatten[4])
min_y = min(cords_flatten[1], cords_flatten[3], cords_flatten[5])
max_y = max(cords_flatten[1], cords_flatten[3], cords_flatten[5])
width = max_x - min_x - width_marker
width1 = int(width * 2)
height = max_y - min_y - width_marker
height1 = int(height * 2)
min_x1 = 0
max_x1 = width1
min_y1 = 0
max_y1 = height1
img = np.zeros((int(height1), int(width1), 3), np.uint8)
color = (0, 204, 0)
cv2.rectangle(img, (0, 0), (width1, height1), color, 3)
char_width = width / 10
char_height = height / 3
char_width1 = width1 / 10
char_height1 = height1 / 3
start_y = min_y + char_height/2 + 10
start_x = min_x + char_width/2 + 10
start_y1 = min_y1 + char_height1 / 2
start_x1 = min_x1 + char_width1 / 2
font = cv2.FONT_HERSHEY_SIMPLEX
for c in charset1:
    centroids_X[ord(c) - 66] = start_x
    centroids_Y[ord(c) - 66] = start_y
    start_x += char_width
    cv2.putText(img, c, (int(start_x1 - 10), int(start_y1)), font, 1, color, 1, cv2.LINE_AA)
    start_x1 += char_width1
start_y += char_height
start_x = min_x + char_width + 10
start_y1 += char_height1
start_x1 = min_x1 + char_width1
for c in charset2:
    centroids_X[ord(c) - 66] = start_x
    centroids_Y[ord(c) - 66] = start_y
    start_x += char_width
    cv2.putText(img, c, (int(start_x1 - 10), int(start_y1)), font, 1, color, 1, cv2.LINE_AA)
    start_x1 += char_width1
start_y += char_height
start_x = min_x + 2 * char_width + 10
start_y1 += char_height1
start_x1 = min_x1 + (2 * char_width1)
for c in charset3:
    centroids_X[ord(c) - 66] = start_x
    centroids_Y[ord(c) - 66] = start_y
    start_x += char_width
    cv2.putText(img, c, (int(start_x1 - 10), int(start_y1)), font, 1, color, 1, cv2.LINE_AA)
    start_x1 += char_width1
cv2.imshow("KeyBoard", img)
cv2.imwrite('KeyBoard.png', img)
cv2.waitKey(200)


def draw_circle(x, y, img):
    x_0 = x.astype(int)
    y_0 = y.astype(int)
    for w in range(x.shape[0]):
        cv2.circle(img, (x_0[w], y_0[w]), 3, (0, 255, 0), -1)
    cv2.imshow('image', img)
    cv2.waitKey(100)
    cv2.destroyAllWindows()


def draw_points(img, image_name, points):
    img_1 = np.copy(img)
    cv2.polylines(img_1, np.int32([points]), False, (0, 255, 0), thickness=3)
    cv2.imshow("gesture", img_1)
    cv2.waitKey(1000)
    cv2.imwrite(image_name, img_1)
    cv2.destroyAllWindows()


def generate_sample_points(points_X, points_Y):
    '''Generate 100 sampled points for a gesture.
    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template computationally.
    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.
    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''
    sample_points_X, sample_points_Y = [], []
    # ediff1d gives difference between consecutive elements of the array
    # we find the distance between coordinates and find the cumulative sum
    distance = np.cumsum(np.sqrt(np.ediff1d(points_X, to_begin=0) ** 2 + np.ediff1d(points_Y, to_begin=0) ** 2))
    # basically when words like mm or ii have no path / little path, use centroid
    if (distance[-1] == 0):
        for i in range(100):
            sample_points_X.append(points_X[0])
            sample_points_Y.append(points_Y[0])
    else:
        # get the proportion of line segments
        distance = distance / distance[-1]
        # scale the points to get linear interpolations along the path
        fx, fy = interp1d(distance, points_X), interp1d(distance, points_Y)
        # generate 100 equidistant points on normalized line
        alpha = np.linspace(0, 1, 100)
        # use the interpolation function to translate from normalized to real plane
        x_regular, y_regular = fx(alpha), fy(alpha)
        sample_points_X = x_regular.tolist()
        sample_points_Y = y_regular.tolist()

    return sample_points_X, sample_points_Y


def normalizeSamples(sample_X, sample_Y):
    L = 1
    sample_Xnorm, sample_Ynorm = [], []

    width = max(sample_X) - min(sample_X)
    height = max(sample_Y) - min(sample_Y)

    if (width == 0 and height == 0):
        s = 0
    else:
        s = L / (max(width, height))

    sample_Xnorm = [s * x for x in sample_X]
    sample_Ynorm = [s * y for y in sample_Y]

    centerX = sum(sample_Xnorm) / len(sample_Xnorm)
    centerY = sum(sample_Ynorm) / len(sample_Ynorm)

    sample_Xnorm = [x - centerX for x in sample_Xnorm]
    sample_Ynorm = [y - centerY for y in sample_Ynorm]

    return sample_Xnorm, sample_Ynorm


# Pre-sample every template and get normalized templates
template_sample_points_X, template_sample_points_Y = [], []
template_sample_points_Xnorm, template_sample_points_Ynorm = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)
    Xnorm, Ynorm = normalizeSamples(template_sample_points_X[i], template_sample_points_Y[i])
    template_sample_points_Xnorm.append(Xnorm)
    template_sample_points_Ynorm.append(Ynorm)


def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y,
               template_sample_points_Xnorm, template_sample_points_Ynorm):
    '''Do pruning on the dictionary of 10000 words.
    In this function, we use the pruning method described in the paper (or any other method you consider it reasonable)
    to narrow down the number of valid words so that the ambiguity can be avoided to some extent.
    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10d000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
    ADDED THESE
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Xnorm: 2D list, containing normalized X-axis values of every template (10d000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Ynorm: 2D list, containing normalized Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :return:
        valid_words: A list of valid words after pruning.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        I ADDED THESE
        valid_template_sample_points_Xnorm: 2D list, the corresponding normalized X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Ynorm: 2D list, the corresponding normalized Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    gesture_Xnorm, gesture_Ynorm = normalizeSamples(gesture_points_X, gesture_points_Y)

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y, valid_template_sample_points_Xnorm, valid_template_sample_points_Ynorm = [], [], [], [], []
    # TODO: Set your own pruning threshold
    threshold = 0.15

    # TODO: Do pruning (12 points)
    for i in range(10000):
        if (abs(template_sample_points_Xnorm[i][0] - gesture_Xnorm[0]) < threshold and abs(
                template_sample_points_Ynorm[i][0] - gesture_Ynorm[0]) < threshold):  # compare first step
            if (abs(template_sample_points_Xnorm[i][-1] - gesture_Xnorm[-1]) < threshold and abs(
                    template_sample_points_Ynorm[i][-1] - gesture_Ynorm[-1]) < threshold):  # compare last step
                valid_words.append(words[i])
                valid_template_sample_points_X.append(template_sample_points_X[i])  # these are not normalized
                valid_template_sample_points_Y.append(template_sample_points_Y[i])  # these are not normalized
                valid_template_sample_points_Xnorm.append(template_sample_points_Xnorm[i])  # these are normalized
                valid_template_sample_points_Ynorm.append(template_sample_points_Ynorm[i])  # these are normalized

    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y, valid_template_sample_points_Xnorm, valid_template_sample_points_Ynorm


def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_Xnorm,
                     valid_template_sample_points_Ynorm):
    '''Get the shape score for every valid word after pruning.
    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.
    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    I CHANGED THESE PARAMETERS - SINCE WE ARE USING NORMALIZED VALUES
    :param valid_template_sample_points_Xnorm: 2D list, containing normalized X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Ynorm: 2D list, containing normalized Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :return:
        A list of shape scores.
    '''

    gesture_Xnorm, gesture_Ynorm = normalizeSamples(gesture_sample_points_X, gesture_sample_points_Y)

    shape_scores = [0] * len(valid_template_sample_points_Xnorm)

    # TODO: Calculate shape scores (12 points)
    for i in range(len(valid_template_sample_points_Xnorm)):
        for j in range(100):
            shape_scores[i] += math.sqrt((gesture_Xnorm[j] - valid_template_sample_points_Xnorm[i][j]) ** 2 + (
                        gesture_Ynorm[j] - valid_template_sample_points_Ynorm[i][j]) ** 2) / 100

    return shape_scores


def isNotZeroD(gestureX, gestureY, templateSampleX, templateSampleY):
    radius = 30

    for i in range(100):
        d = math.sqrt((min(gestureX, key=lambda x: abs(x - templateSampleX[i]))) ** 2 + (
            min(gestureY, key=lambda y: abs(y - templateSampleY[i]))) ** 2)
        if (d - radius > 0):
            return True

    return False


def getBeta(gestureX, gestureY, templateSampleX, templateSampleY, sampleNum):
    if (not isNotZeroD(gestureX, gestureY, templateSampleX, templateSampleY) and not isNotZeroD(templateSampleX,
                                                                                                templateSampleY,
                                                                                                gestureX, gestureY)):
        return 0

    return math.sqrt((gestureX[sampleNum] - templateSampleX[sampleNum]) ** 2 + (
                gestureY[sampleNum] - templateSampleY[sampleNum]) ** 2)


def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X,
                        valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.
    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.
    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :return:
        A list of location scores.
    '''

    alpha = [0] * 100
    for i in range(100):
        alpha[i] = (math.ceil(abs(50.5 - i - 1))) / 2550

    location_scores = [0] * len(valid_template_sample_points_X)

    # TODO: Calculate location scores (12 points)
    for i in range(len(valid_template_sample_points_X)):
        temp = 0
        templateX = valid_template_sample_points_X[i]
        templateY = valid_template_sample_points_Y[i]
        for j in range(100):
            temp += alpha[j] * getBeta(gesture_sample_points_X, gesture_sample_points_Y, templateX, templateY, j) / 100
        location_scores[i] = temp

    return location_scores


def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.5
    # TODO: Set your own location weight
    location_coef = 1 - shape_coef
    for i in range(len(location_scores)):
        integration_scores.append(shape_coef * shape_scores[i] + location_coef * location_scores[i])
    return integration_scores


def get_best_word(valid_words, integration_scores):
    '''Get the best word.
    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.
    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
    best_word = ''
    final_score = float('inf')
    # TODO: Set your own range.
    n = 3
    # TODO: Get the best word (12 points)

    dictionary = dict(zip(valid_words, integration_scores))
    filtered = dict(sorted(dictionary.items(), key=itemgetter(1))[:n])
    # print(filtered)

    for word, int_score in filtered.items():
        if (int_score * (1 - probabilities[word]) < final_score):
            final_score = int_score * (1 - probabilities[word])
            best_word = word

    if (best_word == ''):
        return "No best word found"

    return best_word


@app.route("/")
def init():
    return render_template('index.html')


def cap_vid():
    img1 = cv2.imread('KeyBoard.png')
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    while True:
        _, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow("frame", frame)
    index_tip = []
    _, frame = cap.read()
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    aspect_ratio = frameWidth / frameHeight
    inHeight = 368
    inWidth = int(((aspect_ratio * inHeight) * 8) // 8)
    while cap.isOpened():
        hasFrame, frame = cap.read()
        # frameCopy = np.copy(frame)
        if not hasFrame:
            break
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                        (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
        # Empty list to store the detected keypoints
        points = []
        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]
            probMap = cv2.resize(probMap, (frameWidth, frameHeight))
            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            if prob > threshold:
                if i == 8:
                    index_tip.append(point)
                    if len(index_tip) > 1:
                        prev_x = (int(index_tip[len(index_tip) - 2][0]) - min_x) * 2
                        prev_y = (int(index_tip[len(index_tip) - 2][1]) - min_y) * 2
                        curr_x = (int(index_tip[len(index_tip) - 1][0]) - min_x) * 2
                        curr_y = (int(index_tip[len(index_tip) - 1][1]) - min_y) * 2
                        cv2.line(img1, (prev_x, prev_y), (curr_x, curr_y), color, 2)
                        cv2.imshow("KeyBoard", img1)
                        cv2.imshow("frame", frame)
                    else:
                        cv2.circle(img1, (int(point[0] - min_x) * 2, int(point[1] - min_y) * 2), 3, color, 3)
                        cv2.imshow("KeyBoard", img1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return index_tip


def capture_video():
    cap = cv2.VideoCapture(0)
    img1 = cv2.imread('KeyBoard.png')
    while True:
        _, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow("frame", frame)
    index_tip = []
    has_frame, frame = cap.read()
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    aspect_ratio = frame_width / frame_height
    in_height = 368
    in_width = int(((aspect_ratio * in_height)*8)//8)
    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break
        inp_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (in_width, in_height), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inp_blob)
        output = net.forward()
        # confidence map of corresponding body's part.
        prob_map = output[0, 8, :, :]
        prob_map = cv2.resize(prob_map, (frame_width, frame_height))
        # Find global maximum of the probMap.
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
        if prob > threshold:
            index_tip.append(point)
            if len(index_tip) > 1:
                prev_x = (int(index_tip[len(index_tip) - 2][0]) - max_x) * -4
                prev_y = (int(index_tip[len(index_tip) - 2][1]) - max_y) * -4
                curr_x = (int(index_tip[len(index_tip) - 1][0]) - max_x) * -4
                curr_y = (int(index_tip[len(index_tip) - 1][1]) - max_y) * -4
                cv2.line(img1, (prev_x, prev_y), (curr_x, curr_y), color, 2)
                cv2.imshow("KeyBoard", img1)
            else:
                cv2.circle(img1, (int(index_tip[0][0]), int(index_tip[0][1])), 3, color, 3)
                cv2.imshow("KeyBoard", img1)
        cv2.imshow("frame", frame)
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        aspect_ratio = frame_width / frame_height
        in_height = 368
        in_width = int(((aspect_ratio * in_height) * 8) // 8)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return index_tip


#draw_circle(centroids_X, centroids_Y, image)
q = 0
cap = cv2.VideoCapture(r'C:\Users\fathi\Downloads\W2.mp4')
while True:
    start_time = time.time()
    success, image = cap.read()
    index_tip = []
    img1 = cv2.imread('KeyBoard.png')
    img2 = image.copy()

    image_height, image_width, _ = image.shape

    draw_circle(centroids_X, centroids_Y, image)
    i = 0
    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if i%10 != 0:
                i += 1
                continue
            #image = image[100:500, 700:1200]
            try:
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                # Draw the hand annotations on the image.
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                point = np.zeros(2)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        point[0] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
                        point[1] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
                        print(
                                 f'Index finger tip coordinates: (',
                                 f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                                 f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                             )
                        o = 0
                        res = printKclosest(centroids_X, 26, point[1], 3)
                        if abs(centroids_Y[res[0]] - point[0]) > abs(centroids_Y[res[1]] - point[0]):
                            if abs(centroids_Y[res[1]] - point[0]) < abs(centroids_Y[res[2]] - point[0]):
                                o = 1
                        elif abs(centroids_Y[res[0]] - point[0]) > abs(centroids_Y[res[2]] - point[0]):
                                o = 2
                        index_tip.append(centroids_X[res[o]], centroids_Y[res[o]])
                        #index_tip.append(point)
                        if len(index_tip) > 1:
                            prev_x = (int(index_tip[len(index_tip) - 2][0]) - min_x) * 2
                            prev_y = (int(index_tip[len(index_tip) - 2][1]) - min_y) * 2
                            curr_x = (int(index_tip[len(index_tip) - 1][0]) - min_x) * 2
                            curr_y = (int(index_tip[len(index_tip) - 1][1]) - min_y) * 2
                            cv2.line(img1, (prev_x, prev_y), (curr_x, curr_y), color, 2)
                            cv2.imshow("KeyBoard", img1)
                            cv2.imshow("frame", image)
                            cv2.waitKey(20)
                        else:
                            cv2.circle(img1, (int(point[0] - min_x) * 2, int(point[1] - min_y) * 2), 3, color, 3)
                            cv2.imshow("KeyBoard", img1)
                            cv2.waitKey(20)
                            cv2.waitKey(20)
            except:
                break
        data = np.array(index_tip)
        draw_points(img2, "gesture" + str(q) + ".png", data)
        gesture_points_X = []
        gesture_points_Y = []
        for i in range(len(data)):
            gesture_points_X.append(data[i][0])
            gesture_points_Y.append(data[i][1])

        gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)

        valid_words, valid_template_sample_points_X, valid_template_sample_points_Y, valid_template_sample_points_Xnorm, valid_template_sample_points_Ynorm = do_pruning(
            gesture_sample_points_X, gesture_sample_points_Y, template_sample_points_X, template_sample_points_Y,
            template_sample_points_Xnorm, template_sample_points_Ynorm)

        shape_scores = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y,
                                        valid_template_sample_points_Xnorm, valid_template_sample_points_Ynorm)

        location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y,
                                              valid_template_sample_points_X, valid_template_sample_points_Y)

        integration_scores = get_integration_scores(shape_scores, location_scores)

        best_word = get_best_word(valid_words, integration_scores)
        imh = cv2.imread("gesture" + str(q) + ".png")
        result_gesture = cv2.putText(imh, best_word, (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
        cv2.imshow('result_gesture', result_gesture)
        cv2.imwrite("result_gesture" + str(q) + ".png", result_gesture)
        end_time = time.time()
        print('{"best_word":"' + best_word + '", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}')
        key = cv2.waitKey(1000)
        cv2.destroyAllWindows()
        break


if __name__ == "__main__":
    app.run()
