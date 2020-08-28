import numpy as np
import logging
import cv2

from mri_project.contour_ops import get_muscle_contours

def get_contour_features(cnt):
    area = cv2.contourArea(cnt)
    (sx, sy), (rw, rh), angle = cv2.minAreaRect(cnt)
    (cx, cy), cr = cv2.minEnclosingCircle(cnt)
    xm, ym = cnt.reshape(-1, 2).mean(axis=0)
    xmn, ymn = cnt.reshape(-1, 2).min(axis=0)
    xmx, ymx = cnt.reshape(-1, 2).max(axis=0)
    return [area/(rw*rh), xmn, ymn, xmx, ymx, cx, cy, cr, sx, sy, area, rw, rh, rw/rh, angle, xm, ym]


def get_contours_features(cnts):
    return np.stack([get_contour_features(cnt)+[len(cnts)] for cnt in cnts])
       
    
def get_contours_features_and_colors(img):
    if len(img.shape)>2:
        img = img.sum(axis=2)
    cnts = get_muscle_contours(img>0)
    labels = []
    for cnt in cnts:
        b, a = cnt.reshape(-1 ,2).T
        labels.append(img[(a, b)].mean())
    t = get_contours_features(cnts)
    t = (t-t.mean(axis=0, keepdims=True))/(t.std(axis=0, keepdims=True)+1e-9)
    return np.vstack([t.T, np.argsort(t, axis=0).T]).T, np.array(labels, dtype='uint8').reshape(-1, 1)


def binary_to_multilabel(img, model):
    x, y = get_contours_features_and_colors(img)
    #pred = encoder.inverse_transform(model.predict(x))
    pred = model.predict(x)
    if len(np.unique(pred)) != len(pred):
        return
    imt = np.zeros_like(img, dtype='uint8')
    for cnt, color in zip(get_muscle_contours(img), pred):
        color = int(color)
        #print(color, reverse_muscle_colors.get(color))
        cv2.drawContours(imt, [cnt], -1, color, -1)
    return imt