from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
import pandas as pd


project_root = os.path.join(os.path.expanduser('~'), r"C:\Users\fathi\Documents\Fall_20\theseis\pysot-master")

import cv2
import torch
import numpy as np
from glob import glob

pts = []
imgs = []
sys.path.append(project_root)
torch.set_num_threads(1)


def cal_new_size_v2(im_h, im_w, min_size, max_size):
    rate = 1.0 * max_size / im_h
    rate_w = im_w * rate
    if rate_w > max_size:
        rate = 1.0 * max_size / im_w
    tmp_h = int(1.0 * im_h * rate / 16) * 16

    if tmp_h < min_size:
        rate = 1.0 * min_size / im_h
    tmp_w = int(1.0 * im_w * rate / 16) * 16

    if tmp_w < min_size:
        rate = 1.0 * min_size / im_w
    tmp_h = min(max(int(1.0 * im_h * rate / 16) * 16, min_size), max_size)
    tmp_w = min(max(int(1.0 * im_w * rate / 16) * 16, min_size), max_size)

    rate_h = 1.0 * tmp_h / im_h
    rate_w = 1.0 * tmp_w / im_w
    assert min_size <= tmp_h <= max_size
    assert min_size <= tmp_w <= max_size
    return tmp_h, tmp_w, rate_h, rate_w


def generate_data(im, points, min_size, max_size, im_h, im_w):
    # if len(points) > 0:  # some image has no crowd
    #     idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    #     points = points[idx_mask]
    im_h, im_w, rr_h, rr_w = cal_new_size_v2(im_h, im_w, min_size, max_size)
    #im = np.array(im)
    if rr_h != 1.0 or rr_w != 1.0:
        im = cv2.resize(im, (im_w, im_h), cv2.INTER_CUBIC)
        points[0][0] = points[0][0] * rr_w
        points[0][1] = points[0][1] * rr_h

    density_map = gen_density_map_gaussian(im_h, im_w, points, sigma=8)
    return im, points, density_map


def gen_density_map_gaussian(im_height, im_width, points, sigma=4):
    """
    func: generate the density map.
    points: [num_gt, 2], for each row: [width, height]
    """
    density_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = density_map.shape[:2]
    #num_gt = np.squeeze(points).shape[0]
    #if num_gt == 0:
    #    return density_map
    for p in points:
        p = np.round(p).astype(int)
        p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
        gaussian_radius = sigma * 2 - 1
        gaussian_map = np.multiply(
            cv2.getGaussianKernel(int(gaussian_radius * 2 + 1), sigma),
            cv2.getGaussianKernel(int(gaussian_radius * 2 + 1), sigma).T
        )
        x_left, x_right, y_up, y_down = 0, gaussian_map.shape[1], 0, gaussian_map.shape[0]
        # cut the gaussian kernel
        if p[1] < gaussian_radius:
            x_left = gaussian_radius - p[1]
        if p[0] < gaussian_radius:
            y_up = gaussian_radius - p[0]
        if p[1] + gaussian_radius >= w:
            x_right = gaussian_map.shape[1] - (gaussian_radius + p[1] - w) - 1
        if p[0] + gaussian_radius >= h:
            y_down = gaussian_map.shape[0] - (gaussian_radius + p[0] - h) - 1
        gaussian_map = gaussian_map[y_up:y_down, x_left:x_right]
        if np.sum(gaussian_map):
            gaussian_map = gaussian_map / np.sum(gaussian_map)
        density_map[
        max(0, p[0] - gaussian_radius):min(h, p[0] + gaussian_radius + 1),
        max(0, p[1] - gaussian_radius):min(w, p[1] + gaussian_radius + 1)
        ] += gaussian_map
    density_map = density_map / (np.sum(density_map / 1))
    return density_map


df = pd.read_csv('tsd.csv')
init_rect = []


def main():
    global init_rect
    # load config
    first_frame = True
    o = -1
    images = glob(os.path.join(r"C:\Users\fathi\Documents\Fall_20\theseis\examplecropflaseneg", '*.jpg'))
    imgs.extend(images)
    for img in sorted(imgs):
        frame = cv2.flip(cv2.imread(img), 1)
        o += 1
        fn = img.lstrip(
            'C:\\Users\\fathi\\Documents\\Fall_20\\theseis\\examplecropflaseneg\\')
        if first_frame:
            try:
                init_rect = cv2.selectROI('video_name', frame, False, False)
            except:
                exit()
            #try:
            first_frame = False
            path = r"C:\Users\fathi\Documents\Fall_20\theseis\croppednew"
            path1 = r"C:\Users\fathi\Documents\Fall_20\theseis\numpyarraynew"
            crop_img = frame[init_rect[1]: init_rect[1] + init_rect[3],
                       init_rect[0]: init_rect[0] + init_rect[2]]
            cv2.imshow("img", frame)
            cv2.rectangle(frame, (init_rect[0], init_rect[1]),
                          (init_rect[0] + init_rect[2], init_rect[1] + init_rect[3]),
                          (0, 255, 0), 3)

            cv2.imshow("frame", frame)
            cv2.waitKey(40)
            st = img#"/home/nutfruit/Documents/Fathima599_Fall20/AllVideos/Minh-images/test/TouchDetected/" + fn
            df_rw = df[df['Name'].isin([st])]
            if df_rw.empty:
                density_map = np.zeros([256, 256, 1])
                im_save_path = fn
                dm_save_path = im_save_path.replace('.jpg', '_densitymap.npy')
                np.save(os.path.join(path1, dm_save_path), density_map)
                dim = (256, 256)

                # resize image
                resized = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(path, fn), resized)
                continue
            #Not Detected Frames only
            a = df_rw.iloc[0]['Point0']
            b = df_rw.iloc[0]['Point1']
            if not (init_rect[0] < a < init_rect[0] + init_rect[2] and init_rect[1] < b < init_rect[1] + init_rect[3]):
                density_map = np.zeros([256, 256, 1])
                im_save_path = fn
                dm_save_path = im_save_path.replace('.jpg', '_densitymap.npy')
                np.save(os.path.join(path1, dm_save_path), density_map)
            dim = (256, 256)
            #
            #     # resize image
            resized = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(path, fn), resized)
            continue
            new_p = np.zeros((1, 2))
            new_p = new_p.astype(np.float)
            new_p[0][0] = a-init_rect[0]
            new_p[0][1] = b-init_rect[1]
            image_height, image_width, _ = crop_img.shape
            im, point, density_map = generate_data(crop_img, new_p, 256, 256, image_height, image_width)
            im_save_path = fn
            dm_save_path = im_save_path.replace('.jpg', '_densitymap.npy')
            np.save(os.path.join(path1, dm_save_path), density_map)
            cv2.imwrite(os.path.join(path, fn), im)

            # except:
            #     continue
        else:
            f1 = frame.copy()
            cv2.rectangle(f1, (init_rect[0], init_rect[1]),
                          (init_rect[0] + init_rect[2], init_rect[1] + init_rect[3]),
                          (0, 255, 0), 3)
            cv2.imshow("img", f1)
            key = cv2.waitKey(1000000)
            if key == ord("s"):
                #first_frame = True
                init_rect = cv2.selectROI('video_name', frame, False, False)
            elif key == ord("q"):
                first_frame = False
            bbox = init_rect ##list(map(int, outputs['bbox']))
            path = r"C:\Users\fathi\Documents\Fall_20\theseis\croppednew"
            path1 = r"C:\Users\fathi\Documents\Fall_20\theseis\numpyarraynew"
            crop_img = frame[bbox[1]: bbox[1] + bbox[3],
                       bbox[0]: bbox[0] + bbox[2]]
            st = img#"/home/nutfruit/Documents/Fathima599_Fall20/AllVideos/Minh-images/test/TouchDetected/" + fn
            df_rw = df[df['Name'].isin([st])]

            if df_rw.empty:
                density_map = np.zeros([256, 256, 1])
                im_save_path = fn
                dm_save_path = im_save_path.replace('.jpg', '_densitymap.npy')
                np.save(os.path.join(path1, dm_save_path), density_map)
                dim = (256, 256)

                # resize image
                resized = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(path, fn), resized)
                continue
            dim = (256, 256)
            #
            #     # resize image
            resized = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(path, fn), resized)
            # Not Detected Frames only
            a = df_rw.iloc[0]['Point0']
            b = df_rw.iloc[0]['Point1']
            if not (bbox[0] < a < bbox[0] + bbox[2] and bbox[1] < b < bbox[1] + bbox[3]):
                density_map = np.zeros([256, 256, 1])
                im_save_path = fn
                dm_save_path = im_save_path.replace('.jpg', '_densitymap.npy')
                np.save(os.path.join(path1, dm_save_path), density_map)
                dim = (256, 256)

                # resize image
                resized = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(path, fn), resized)
                continue
            new_p = np.zeros((1, 2))
            new_p = new_p.astype(np.float)
            new_p[0][0] = a - init_rect[0]
            new_p[0][1] = b - init_rect[1]
            image_height, image_width, _ = crop_img.shape
            im, point, density_map = generate_data(crop_img, new_p, 256, 256, image_height, image_width)
            im_save_path = fn
            dm_save_path = im_save_path.replace('.jpg', '_densitymap.npy')
            np.save(os.path.join(path1, dm_save_path), density_map)
            cv2.imwrite(os.path.join(path, fn), im)
        cv2.rectangle(frame, (init_rect[0], init_rect[1]),
                          (init_rect[0]+init_rect[2], init_rect[1]+init_rect[3]),
                          (0, 255, 0), 3)


if __name__ == '__main__':
    main()

