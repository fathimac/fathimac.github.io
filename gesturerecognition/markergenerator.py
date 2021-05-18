import cv2 as cv
import numpy as np

# Load the predefined dictionary
dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

# Generate the marker
markerImage = np.zeros((50, 50), dtype=np.uint8)
markerImage = cv.aruco.drawMarker(dictionary, 33, 50, markerImage, 1);

cv.imwrite("marker34.png", markerImage);