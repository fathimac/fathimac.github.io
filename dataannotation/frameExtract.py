import numpy as np
import os
import cv2
import random

filename1 = 'filename'

cap1 = cv2.VideoCapture(filename1)
ret0, frame0 = cap1.read()
frame0 = cv2.flip(frame0, 1)
count1 = 0

min = 0
max = 999
digits = [str(random.randint(min, max)) for i in range(5)]
digits = [(len(str(max))-len(digit))*'0'+digit for digit in digits]
while True:
    ret0, frame0 = cap1.read()
    if not ret0:
        break
    key = cv2.waitKey(1) & 0xff
    cv2.imwrite("SR-%s.jpg" %(str(190)+str(count1).zfill(6)), frame0)  

    count1 += 1    


cap1.release()
cv2.destroyAllWindows()
