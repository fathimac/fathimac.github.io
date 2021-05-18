'''

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
import math
import scipy
from scipy import spatial
import numpy as np
import pandas as pd
import cv2


app = Flask(__name__)


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
    while True:
        ret0, frame0 = cap1.read()
        if ret0:
            frame = np.copy(frame0)
            # converting BGR to GRAYSCALE
            gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(gray, 10, 250)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
            # find contours (i.e. the 'outlines') in the image and initialize the
            # total number of books found
            (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total = 0
            # Display edges in a frame
            # loop over the contours
            approx = np.zeros(4)
            for c1 in cnts:
                # approximate the contour
                peri = cv2.arcLength(c1, True)
                approx = cv2.approxPolyDP(c1, 0.02 * peri, True)

                # if the approximated contour has four points, then assume that the
                # contour is a book -- a book is a rectangle and thus has four vertices
                if len(approx) == 4:
                    print(approx)
                    cv2.drawContours(frame0, [approx], -1, (0, 255, 0), 4)
                    cv2.imshow("Output", frame0)
                    key = cv2.waitKey(0)
                    if key == ord('a'):
                        cap1.release()
                        cv2.destroyAllWindows()
                        return approx, frame
                    total += 1
            # display the output

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap1.release()
    cv2.destroyAllWindows()
    return approx


# Centroids of 26 keys

charset1 = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P']
charset2 = ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L']
charset3 = ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
# header_list = ["x", "y"]
# df=pd.read_csv('alphabet_loc.csv', sep=',', names=header_list, header=None)
# np_arr = df[['x', 'y']].to_numpy()
centroids_X = np.zeros(26)
centroids_Y = np.zeros(26)
# for i in range(0, 26):
#     centroids_X[i] = np_arr[i][0]
#     centroids_Y[i] = np_arr[i][1]
cords, image = get_keyboard()
cords_flatten = cords.flatten()
min_x = min(cords_flatten[0], cords_flatten[2], cords_flatten[4], cords_flatten[6])
max_x = max(cords_flatten[0], cords_flatten[2], cords_flatten[4], cords_flatten[6])
min_y = min(cords_flatten[1], cords_flatten[3], cords_flatten[5], cords_flatten[7])
max_y = max(cords_flatten[1], cords_flatten[3], cords_flatten[5], cords_flatten[7])
width = max_x - min_x
width1 = width * 4
height = max_y - min_y
height1 = height * 4
min_x1 = 0
max_x1 = width1
min_y1 = 0
max_y1 = height1
img = np.zeros((height1, width1, 3), np.uint8)
color = (0, 204, 0)
cv2.rectangle(img, (0, 0), (width1, height1), color, 3)
char_width = width/10
char_height = height/3
char_width1 = width1/10
char_height1 = height1/3
start_y = max_y - char_height/2
start_x = max_x - char_width/2
start_y1 = min_y1 + char_height1/2
start_x1 = min_x1 + char_width1/2
font = cv2.FONT_HERSHEY_SIMPLEX
for c in charset1:
    centroids_X[ord(c) - 66] = start_x
    centroids_Y[ord(c) - 66] = start_y
    start_x -= char_width
    cv2.putText(img, c, (int(start_x1 - 10), int(start_y1)), font, 1, color, 1, cv2.LINE_AA)
    start_x1 += char_width1
start_y -= char_height
start_x = max_x - char_width
start_y1 += char_height1
start_x1 = min_x1 + char_width1
for c in charset2:
    centroids_X[ord(c) - 66] = start_x
    centroids_Y[ord(c) - 66] = start_y
    start_x -= char_width
    cv2.putText(img, c, (int(start_x1 - 10), int(start_y1)), font, 1, color, 1, cv2.LINE_AA)
    start_x1 += char_width1
start_y -= char_height
start_x = max_x - 2 * char_width
start_y1 += char_height1
start_x1 = min_x1 + (2 * char_width1)
for c in charset3:
    centroids_X[ord(c) - 66] = start_x
    centroids_Y[ord(c) - 66] = start_y
    start_x -= char_width
    cv2.putText(img, c, (int(start_x1 - 10), int(start_y1)), font, 1, color, 1, cv2.LINE_AA)
    start_x1 += char_width1
cv2.imshow("KeyBoard", img)
cv2.imwrite('KeyBoard.png', img)
cv2.waitKey(2000)


def draw_circle(x, y, img):
    x_0 = x.astype(int)
    y_0 = y.astype(int)
    for w in range(x.shape[0]):
        cv2.circle(img, (x_0[w], y_0[w]), 3, (0, 255, 0), -1)
    cv2.imshow('image', img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()


class StartEndNode:
    def __init__(self, startX, endX, startY, endY):
        self.startX = startX
        self.startY = startY
        self.endX = endX
        self.endY = endY


class Node:

    # Function to initialise the node object
    def __init__(self, points):
        self.data = points  # Assign data
        self.next = None  # Initialize next as null


# Linked List class contains a Node object
class LinkedList:

    # Function to initialize head
    def __init__(self):
        self.head = None


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
        template_points_X[-1].append(centroids_X[ord(c) - 98])
        template_points_Y[-1].append(centroids_Y[ord(c) - 98])


def draw_points(img, image_name, points):
    img_1 = np.copy(img)
    cv2.polylines(img_1, [points], False, (0, 255, 0), thickness=3)
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
    # TODO: Start sampling (12 points)
    length = 0
    distance_array = []
    for u in range(0, len(points_X) - 1):
        distance = math.sqrt(pow(points_X[u] - points_X[u + 1], 2) + pow(points_Y[u] - points_Y[u + 1], 2))
        distance_array.append(distance)
        length += distance
    index = 0
    leftover = 0
    if length != 0:
        for k in range(0, len(distance_array)):
            if distance_array[k] != 0:
                numpoints = (distance_array[k] / length) * 100
                x_inc = (points_X[k + 1] - points_X[k]) / numpoints
                y_inc = (points_Y[k + 1] - points_Y[k]) / numpoints
                leftover += numpoints - int(numpoints)
                if leftover >= 1:
                    leftover -= 1
                    numpoints += 1
                for l in range(0, int(numpoints)):
                    sample_points_X.append(points_X[k] + l * x_inc)
                    sample_points_Y.append(points_Y[k] + l * y_inc)
                    index += 1
                    if index == 100:
                        return sample_points_X, sample_points_Y
    else:
        for m in range(0, 100):
            sample_points_X.append(points_X[0])
            sample_points_Y.append(points_Y[0])
        return sample_points_X, sample_points_Y
    if index < 100:
        sample_points_X.append(points_X[len(points_X) - 1])
        sample_points_Y.append(points_Y[len(points_X) - 1])
    return sample_points_X, sample_points_Y


# Pre-sample every template
template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)
alpha_0 = np.random.dirichlet(np.ones(100) * 1000, size=1)
alpha_1 = np.sort(alpha_0[0][:50])[::-1]
alpha_2 = np.sort(alpha_0[0][50:])
alpha = np.concatenate((alpha_1, alpha_2))


def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider it reasonable)
    to narrow down the number of valid words so that the ambiguity can be avoided to some extent.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    # TODO: Set your own pruning threshold
    threshold = 20
    # TODO: Do pruning (12 points)

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    template_points = LinkedList()
    start_end_node = StartEndNode(template_sample_points_X[0][0], template_sample_points_X[0][99],
                                  template_sample_points_Y[0][0], template_sample_points_Y[0][99])
    template_points.head = Node(start_end_node)
    current_pointer = template_points.head
    for l in range(1, len(template_sample_points_X)):
        node = Node(StartEndNode(template_sample_points_X[l][0],
                                 template_sample_points_X[l][len(template_sample_points_X[l]) - 1],
                                 template_sample_points_Y[l][0],
                                 template_sample_points_Y[l][len(template_sample_points_Y[l]) - 1]))
        current_pointer.next = node
        current_pointer = node
    current_pointer = template_points.head
    start_x = gesture_points_X[0]
    start_y = gesture_points_Y[0]
    end_x = gesture_points_X[len(gesture_points_X) - 1]
    end_y = gesture_points_Y[len(gesture_points_Y) - 1]
    p = 0
    while current_pointer is not None:
        start_end_node = current_pointer.data
        distance_start = math.sqrt(pow(start_end_node.startX - start_x, 2) + pow(start_end_node.startY - start_y, 2))
        distance_end = math.sqrt(pow(start_end_node.endX - end_x, 2) + pow(start_end_node.endY - end_y, 2))
        if (distance_start < threshold) and (distance_end < threshold):
            valid_template_sample_points_X.append(template_sample_points_X[p])
            valid_template_sample_points_Y.append(template_sample_points_Y[p])
            valid_words.append(words[p])
        current_pointer = current_pointer.next
        p += 1
    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y


def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X,
                     valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''
    gesture_sample_points_X_temp = np.zeros(100)
    gesture_sample_points_Y_temp = np.zeros(100)
    valid_template_sample_points_X_temp = np.zeros(shape=(len(valid_template_sample_points_X), 100))
    valid_template_sample_points_Y_temp = np.zeros(shape=(len(valid_template_sample_points_Y), 100))
    shape_scores = []
    # TODO: Set your own L
    L = 1
    s = 1
    W = max(gesture_sample_points_X) - min(gesture_sample_points_X)
    H = max(gesture_sample_points_Y) - min(gesture_sample_points_Y)
    '''For a single point, the bounding box values can be zero. To handle this case, we are checking if it's zero '''
    if W != 0 or H != 0:
        s = L / max(W, H)
    else:
        s = 1/(gesture_sample_points_X[0]*gesture_sample_points_Y[0])
    for k in range(0, len(gesture_sample_points_X)):
        gesture_sample_points_X_temp[k] = gesture_sample_points_X[k] * s
        gesture_sample_points_Y_temp[k] = gesture_sample_points_Y[k] * s
    for j in range(0, len(valid_template_sample_points_X)):
        W = max(valid_template_sample_points_X[j]) - min(valid_template_sample_points_X[j])
        H = max(valid_template_sample_points_Y[j]) - min(valid_template_sample_points_Y[j])
        if W != 0 or H != 0:
            s = L / max(W, H)
        else:
            s = 1 / (valid_template_sample_points_X[j][0] * valid_template_sample_points_Y[j][0])
        for k in range(0, len(valid_template_sample_points_X[j])):
            valid_template_sample_points_X_temp[j][k] = valid_template_sample_points_X[j][k] * s
            valid_template_sample_points_Y_temp[j][k] = valid_template_sample_points_Y[j][k] * s
    # TODO: Calculate shape scores (12 points)
    for j in range(0, len(valid_template_sample_points_X_temp)):
        shape_score = 0
        for k in range(0, 100):
            u = [gesture_sample_points_X_temp[k], gesture_sample_points_Y_temp[k]]
            v = [valid_template_sample_points_X_temp[j][k], valid_template_sample_points_Y_temp[j][k]]
            shape_score += scipy.spatial.distance.euclidean(u, v)
        shape_score = shape_score / 100
        shape_scores.append(shape_score)
    return shape_scores


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

    location_scores = []
    radius = 15
    ## TODO: Calculate location scores (12 points)
    d_p_q = 0
    D = np.zeros(shape=(len(valid_template_sample_points_X), 2))
    for j in range(0, len(valid_template_sample_points_X)):
        for y in range(0, 100):
            u = [gesture_sample_points_X[y], gesture_sample_points_Y[y]]
            d = []
            for z in range(0, 100):
                v = [valid_template_sample_points_X[j][z], valid_template_sample_points_Y[j][z]]
                d.append(math.sqrt(pow(v[0] - u[0], 2) + pow(v[1] - u[1], 2)))
            d_p_q += (max(min(d) - radius, 0))
        D[j][0] = d_p_q
        d_q_p = 0
        for y in range(0, 100):
            v = [valid_template_sample_points_X[j][y], valid_template_sample_points_Y[j][y]]
            d = []
            for z in range(0, 100):
                u = [gesture_sample_points_X[z], gesture_sample_points_Y[z]]
                d.append(math.sqrt(pow(v[0] - u[0], 2) + pow(v[1] - u[1], 2)))
            d_q_p += (max(min(d) - radius, 0))
        D[j][1] = d_p_q
    for j in range(0, len(valid_template_sample_points_X)):
        x_l = 0
        for k in range(0, 100):
            delta = 0 if D[j][0] == 0 and D[j][1] == 0 else math.sqrt(pow(gesture_sample_points_X[k] -
                                                                  valid_template_sample_points_X[j][k], 2) +
                                                                  pow(gesture_sample_points_Y[k] -
                                                                  valid_template_sample_points_Y[j][k], 2))
            x_l += delta * alpha[k]
        location_scores.append(x_l)
    return location_scores


def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.5
    # TODO: Set your own location weight
    location_coef = 0.5
    for u in range(len(shape_scores)):
        integration_scores.append(shape_coef * shape_scores[u] + location_coef * location_scores[u])
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
    #TODO: Set your own range.
    n = 3
    best_word = []
    #TODO: Get the best word (12 points)
    if len(integration_scores) != 0:
        best_index = sorted(range(len(integration_scores)), key=lambda x: integration_scores[x])[:n]
        for index in range(len(best_index)):
            best_word.append(valid_words[best_index[index]])
    else:
        best_word = ['the']
    return best_word


@app.route("/")
def init():
    return render_template('index.html')


def cap_vid():
    img1 = cv2.imread('KeyBoard.png')
    cap = cv2.VideoCapture(0)
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
    while 1:
        hasFrame, frame = cap.read()
        #frameCopy = np.copy(frame)
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
                        prev_x = (int(index_tip[len(index_tip) - 2][0]) - max_x) * -4
                        prev_y = (int(index_tip[len(index_tip) - 2][1]) - max_y) * -4
                        curr_x = (int(index_tip[len(index_tip) - 1][0]) - max_x) * -4
                        curr_y = (int(index_tip[len(index_tip) - 1][1]) - max_y) * -4
                        cv2.line(img1, (prev_x, prev_y), (curr_x, curr_y), color, 2)
                        cv2.imshow("KeyBoard", img1)
                    else:
                        cv2.circle(img1, (int(point[0] - max_x) * -4, int(point[1]- max_y) * -4), 3, color, 3)
                        cv2.imshow("KeyBoard", img1)
                # Add the point to the list if the probability is greater than the threshold
            #     points.append((int(point[0]), int(point[1])))
            #
            # else:
            #     points.append(None)
        # Draw Skeleton
        # for pair in POSE_PAIRS:
        #     partA = pair[0]
        #     partB = pair[1]
        #     if points[partA] and points[partB]:
        #         cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2, lineType=cv2.LINE_AA)
        #         cv2.circle(frame, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        #         cv2.circle(frame, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        # cv2.imshow("frame", frame)
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


draw_circle(centroids_X, centroids_Y, image)
q = 0
while True:
    start_time = time.time()
    data = np.array(cap_vid())
    draw_points(image, "gesture" + str(q) + ".png", data)
    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i][0])
        gesture_points_Y.append(data[i][1])

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_sample_points_X,
                                                                                             gesture_sample_points_Y,
                                                                                             template_sample_points_X,
                                                                                             template_sample_points_Y)
    shape_scores = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X,
                                    valid_template_sample_points_Y)
    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y,
                                          valid_template_sample_points_X, valid_template_sample_points_Y)
    integration_scores = get_integration_scores(shape_scores, location_scores)
    best_word = get_best_word(valid_words, integration_scores)
    end_time = time.time()
    best = best_word[0]
    imh = cv2.imread("gesture" + str(q) + ".png")
    result_gesture = cv2.putText(imh, best, (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
    cv2.imshow('result_gesture', result_gesture)
    cv2.imwrite("result_gesture" + str(q) + ".png", result_gesture)
    print('{"best_word":"' + best + '", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}')
    key = cv2.waitKey(0)
    if key == ord('r'):
        cv2.destroyAllWindows()
        break
    q += 1


# if __name__ == "__main__":
#     app.run()
