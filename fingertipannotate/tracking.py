from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import imutils
import numpy as np
import joblib
import os
import sys

project_root = os.path.join(os.path.expanduser('~'), r"C:\Users\fathi\Documents\Fall_20\theseis\pysot-master")

import argparse
import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
pts = []
imgs = []
sys.path.append(project_root)

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()


def draw_roi(event, x, y, flags, param):
    img = cv2.imread('first.png')
    img2 = img.copy()

    if event == cv2.EVENT_LBUTTONDOWN:  # Left click, select point
        pts.append((x, y))

    # if event == cv2.EVENT_RBUTTONDOWN:  # Right click to cancel the last selected point
    #    pts.pop()

    if event == cv2.EVENT_RBUTTONDOWN:
        mask = np.zeros(img.shape, np.uint8)
        points = np.array(pts, np.int32)
        points = points.reshape((-1, 1, 2))

        mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
        mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))  # for ROI
        mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))  # for displaying images on the desktop

        show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)

        #cv2.imshow("mask", mask2)
        #cv2.imshow("show_img", show_image)

        ROI = cv2.bitwise_and(mask2, img)
        #cv2.imshow("ROI", ROI)
        cv2.waitKey(0)
        #return ROI

    if len(pts) > 0:
        # Draw the last point in pts
        cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)

    if len(pts) > 1:
        #
        for i in range(len(pts) - 1):
            cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)  # x ,y is the coordinates of the mouse click place
            cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)

    cv2.imshow('image', img2)


# Create images and windows and bind windows to callback function


def get_frames(video_name):
   
    images = glob(os.path.join(r"C:\Users\fathi\Documents\Fall_20\theseis\test\TouchDetected", '*.jpg'))
    imgs.append(images)
    #images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    for img in images:
        frame = cv2.imread(img)
        yield frame




def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    o = -1
    for frame in get_frames(args.video_name):
        o += 1
        if first_frame:
            try:
                
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            try:
                fn = imgs[0][o].lstrip(
                    'C:\\Users\\fathi\\Documents\\Fall_20\\theseis\\test\\TouchDetected\\')
                path = r"C:\Users\fathi\Documents\Fall_20\theseis\test\cropped\TouchDetected"
                path1 = r"C:\Users\fathi\Documents\Fall_20\theseis\test\cropped\TouchDetected"
             
                crop_img = frame[init_rect[1]: init_rect[1] + init_rect[3],
                           init_rect[0]: init_rect[0] + init_rect[2]]
                dim = (256, 256)

                # resize image
                resized = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(path, fn), resized)
                dim = (512, 512)

                # resize image
                # resized1 = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                # cv2.imwrite(os.path.join(path1, fn), resized1)
            except:
                continue
            #first_frame = False
        else:

            outputs = tracker.track(frame)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                # center = (bbox[1] + (bbox[3] // 2), bbox[0] + (bbox[2] // 2))
                # if bbox[2] > bbox[3]:
                #     toCoverWidth = 1.25 * bbox[2]
                # else:
                #     toCoverWidth = 1.25 * bbox[3]
                # bbox1 = np.array(
                #     [center[1] - toCoverWidth // 2, center[0] - toCoverWidth // 2, toCoverWidth, toCoverWidth])

                #bbox1 = bbox.astype(dtype=np.int)
                #print("otherscm: " + fn)
                crop_img = frame[bbox[1]: bbox[1] + bbox[3],
                           bbox[0]: bbox[0] + bbox[2]]
                fn = imgs[0][o].lstrip(
                    'C:\\Users\\fathi\\Documents\\Fall_20\\theseis\\test\\TouchDetected\\')
                path = r"C:\Users\fathi\Documents\Fall_20\theseis\test\cropped\TouchDetected"
                path1 = r"C:\Users\fathi\Documents\Fall_20\theseis\test\cropped\TouchDetected"
                try:
                    dim = (256, 256)
                    # resize image
                    resized = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(path, fn), resized)

                    dim = (512, 512)

                    # resize image
                    # resized1 = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                    # cv2.imwrite(os.path.join(path1, fn), resized1)
                except:
                    continue
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)

            cv2.imshow(video_name, frame)
            cv2.waitKey(40)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("s"):
                first_frame = True

if __name__ == '__main__':
    main()
