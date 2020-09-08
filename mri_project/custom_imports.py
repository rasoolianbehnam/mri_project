import numpy as np
import shutil
import glob
import random
import dask
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
import pandas as pd
from dask.distributed import Client, LocalCluster
from dask import delayed, compute
from dask import bag, array as da
import joblib
import scipy.stats as st
from scipy.ndimage.filters import convolve
from scipy.ndimage import zoom
from multiprocessing import Pool
from matplotlib.pyplot import subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder



# def plots(*args, **kwargs):
#     fig, axes = plt.subplots(*args, **kwargs)
#     return axes
# 
# 
# #pylab.rcParams['figure.figsize'] = (10, 9)
# def threshold(img, low, high, label):
#     w, h, d = img.shape
#     masks = []
#     for i in range(3):
#         masks.append(img[:, :, i] <= high[i])
#         masks.append(img[:, :, i] >= low[i])
#     #masks.append(img[:, :, 2] > 200)
#     mask = np.ones((w, h))
#     for m in masks:
#         mask *= m   
#     return(mask * label)
# def preprocess_label(label, shape):
#     label = cv2.resize(label, shape)
#     m1 = threshold(label, [0, 0, 0], [1, 1, 255], 1)
#     m2 = threshold(label, [240, 0, 0], [255, 30, 255], 2)
#     m3 = threshold(label, [255, 255, 255], [255, 255, 255], 3)
#     #m4 = threshold(label, [0, 0, 240], [60, 255, 255], 4)
#     m = m1 + m2 + m3 #+ m4
#     m = cv2.erode(m, np.ones((3, 3)))
#     return m
# 
# 
# 
# def gkern2d(kernlen=21, nsig=3):
#     """
#     @param: kernlen: size of the 2d kernel
#     @nsig: sigma for the 2d gaussian distribution
#     returns: a gaussian kernel scaled so that the middle point set to 1
#     """
#     """Returns a 2D Gaussian kernel."""
#     x = np.linspace(-nsig, nsig, kernlen+1)
#     kern1d = np.diff(st.norm.cdf(x))
#     kern2d = np.outer(kern1d, kern1d)
#     return kern2d/kern2d.max()
# def get_fucked(y, x=None):
#     if x is not None:
#         x = np.mean(x, axis=2)
#     _1, cnts, _2 = cv2.findContours(y.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
#     imt = np.zeros_like(y)
#     cv2.drawContours(imt, cnts, -1, 255, 10)
#     if x is not None:
#         assert x.shape[:2] == y.shape[:2]
#     X = []
#     for cnt in cnts:
#         cnt = cnt.reshape(-1, 2)
#         X.append(np.array([cnt.min(axis=0), cnt.max(axis=0), cnt.mean(axis=0)]))
#     _ = np.array(X).reshape(-1, 6)
#     return np.concatenate([_, np.argsort(_, axis=0)], axis=1)
#     #plt.imshow(get_torso_outline((x, y)))
# 
# def filter_contour(img=None):
#     def _filter_contour(x):
#         _, rct_w, _, rct_h = cv2.boundingRect(x)
#         cnt_area = cv2.contourArea(x)
#         area_cnd = True
#         if img is not None:
#             area_cnd = (cnt_area / len(img.reshape(-1))) > .3
#         return (rct_h / (rct_w+1e-8) < 10)  & area_cnd
#     return _filter_contour
# def get_torso(img, block_sizes):
#     if len(img.shape) > 2:
#         grey_img = img.mean(axis=2).astype('uint8')
#     else:
#         grey_img = img.astype('uint8')
#     for block_size in block_sizes:
#         thrshldd_img = cv2.adaptiveThreshold(grey_img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 0)
#         #thrshldd_img = cv2.adaptiveThreshold(grey_img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 0)
#         erdd_img = cv2.erode(thrshldd_img, np.ones((5,5)))
#         _, cnts, _ = cv2.findContours(erdd_img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#         cnts = sorted(filter(filter_contour(thrshldd_img), cnts), key=lambda x: -len(x))
#         if len(cnts):
#             break
#     return cnts, thrshldd_img
# 
# def get_torso_outline(x):
#     orig, masked = x
#     print(tuple(i.shape for i in x))
#     cnts, thrshldd_img = get_torso(orig, 
#                                #range(501, 50, -10)
#                                range(501, 50, -10)
#                               )
#     if len(cnts) == 0:
#         return
#     img_to_drw = np.zeros_like(thrshldd_img)
#     _ = cv2.drawContours(img_to_drw, cnts[:1], -1, 1, 40)
#     scnd_img_to_drw = np.zeros_like(thrshldd_img)
#     _, cnts, _ = cv2.findContours(img_to_drw.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     cnts = sorted(cnts, key = lambda x: -len(x))
#     _ = cv2.drawContours(scnd_img_to_drw, cnts, 0, 1, -11)
#     kernel = np.ones((10, 10))
#     out_and_masked = scnd_img_to_drw*thrshldd_img
#     rszd_out_and_masked = cv2.resize(out_and_masked, masked.shape[::-1])
#     out = cv2.dilate((masked>0).astype('uint8') | rszd_out_and_masked, kernel)
#     return out
# 
# def resize_image(x, width):
#     w, h, *_ = x.shape
#     new_shape = h*width//w, width
#     return cv2.resize(x, new_shape)
# 
# def resize_images(args, dsize):
#     return tuple(cv2.resize(x, dsize) for x in args)
# 
# def resize_image(args, dsize):
#     return cv2.resize(args, dsize)
# 
# 
# # 1. Setting mask margins to a certain value so they are never picked for training or test
# def mark_margins(y, n_size):
#     x = y.copy()
#     m = n_size//2+1
#     x[:m] = 255
#     x[-m:] = 255
#     x[:, :m] = 255
#     x[:, -m:] = 255
#     return x
# 
# # zooming in as much as possigle on the area of interes
# # which is teh traced muscles
# def crop_image(args, neigh_size):
#     m = neigh_size//2
#     x, y = args
#     pts = np.where(y==1)
#     w, h = y.shape
#     x_mn = max(0, np.min(pts[0])-m)
#     x_mx = min(w, np.max(pts[0])+m)
#     y_mn = max(0, np.min(pts[1])-m)
#     y_mx = min(h, np.max(pts[1])+m)
#     #r = int((y_mx - y_mn)/(x_mx-x_mn) * resize)
#     return x[x_mn:x_mx, y_mn:y_mx], y[x_mn:x_mx, y_mn:y_mx]
# 
# def normalize_image(img):
#     img = img.astype('uint8')
#     hist,bins = np.histogram(img.flatten(),256,[0,256])
# 
#     cdf = hist.cumsum()
#     cdf_m = np.ma.masked_equal(cdf,0)
#     cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
#     cdf = np.ma.filled(cdf_m,0).astype('uint8')
#     return cdf[img]
# def get_shape_ratios(args):
#     x, y = args
#     r1 = x.shape[1]/x.shape[0]
#     r2 = y.shape[1]/y.shape[0]
#     return np.abs(r1-r2) < .01
# 
# def get_neighbors(xy, neigh_size, step):
#     m = neigh_size//2
#     if isinstance(xy, tuple):
#         x, y = xy
#         # change later
#         x = (x-x.mean())/x.std()
#         w, h, *_ = x.shape
#         X = []
#         Y = []
#         for i in range(m, w-m, step):
#             for j in range(m, h-m, step):
#                 X.append(x[i-m:i+m, j-m:j+m])
#                 Y.append(y[i, j])
#         return np.array(X), np.array(Y)
#     else:
#         x = xy
#         x = (x-x.mean())/x.std()
#         w, h, *_ = x.shape
#         X = []
#         for i in range(m, w-m, step):
#             for j in range(m, h-m, step):
#                 X.append(x[i-m:i+m, j-m:j+m])
#         return np.array(X)
#     
# def get_neighbors_masked(xyz, neigh_size, step):
#     m = neigh_size//2
#     if isinstance(xyz, tuple):
#         x, y, z= xyz
#         # change later
#         x = (x-x.mean())/x.std()
#         w, h, *_ = x.shape
#         X = []
#         Y = []
#         for i, j in zip(*np.where(z)):
#             X.append(x[i-m:i+m, j-m:j+m])
#             Y.append(y[i, j])
#         return np.array(X), np.array(Y)
#     else:
#         x = xyz
#         x = (x-x.mean())/x.std()
#         w, h, *_ = x.shape
#         X = []
#         for i, j in zip(*np.where(z)):
#             X.append(x[i-m:i+m, j-m:j+m])
#         return np.array(X)
#     
# def resize_float(x, dsize):
#     if x is None:
#         return None
#     return zoom(x, (i/j for i,j in zip(dsize, x.shape)), mode='nearest')
# 
# def resize_int(x, dsize):
#     if x is not None:
#         return cv2.resize(x.astype('uint8'), dsize, interpolation=cv2.INTER_NEAREST)
# 
# def prepare_binary_mask_for_multi_label_prediction(x):
#     if len(x.shape)>2:
#         x = x.sum(axis=2)
#     shape = np.array(x.shape).reshape(-1, 2)
#     print(np.unique(x))
#     y = (x > 0).astype('uint8')
#     _, cnts, _ = cv2.findContours(y.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
#     imt = np.zeros_like(y)
#     cv2.drawContours(imt, cnts, -1, 255, 10)
#     X = []
#     Y = []
#     C1 = {}
#     C2 = {}
#     I = []
#     for cnt in cnts:
#         cnt = cnt.reshape(-1, 2)
#         imt2 = np.zeros_like(y)
#         cv2.drawContours(imt2, [cnt], -1, 1, -1)
#         Y.append(x[np.where(imt2)].mean())
#         X.append(np.array([cnt.min(axis=0), cnt.max(axis=0), cnt.mean(axis=0)]))
#         ff = np.array(np.where(imt2)).T
#         mn, mx = ff.min(axis=0, keepdims=True), ff.max(axis=0, keepdims=True)
#         C1[Y[-1]] = (ff-mn)/mx
#         C2[Y[-1]] = ff 
#         
#         I.append(imt2)
#     X = np.array(X).reshape(-1, 6)
#     mn = X[:, :2].min()+1e-9
#     mx = X[:, :2].max()+1e-8
#     X = (X - mn)/(mx-mn)
#     X_ordered = np.argsort(X, axis=0)
#     return np.concatenate([X, X_ordered], axis=1), np.array(Y).reshape(-1, 1), C2, C1, np.array(I)
# 
# B = {0, 5, 10, 15, 19, 20, 26, 29, 35, 37, 43, 45, 51, 55, 57, 60, 69, 73, 76, 82, 86, 90, 93, 99, 102,
#  104, 107, 111, 113, 118, 121, 125, 128, 138, 141, 144, 151, 154, 156, 160, 164, 169, 178, 186, 188,
#  192, 201, 206, 210, 214, 219, 221, 226, 230, 234, 237, 243, 249, 255, 257, 261, 267, 271, 272, 277, 280,
#  285, 291, 295, 296, 310, 313, 319, 323, 325, 330, 332, 338, 343, 347, 349, 355, 358, 362, 366, 371, 374,
#  376, 381, 386, 388, 393, 399, 405, 411, 413, 418, 420, 424, 431, 435, 439, 446, 448, 465, 467, 468, 472,
#  476, 482, 484, 624, 633, 636, 640, 645, 650, 659, 663, 664, 669, 672, 678, 680, 684, 689, 693, 696, 702, 706,
#  710, 712, 721, 724, 728, 731, 733, 739, 740, 747, 751, 752, 757, 762, 764, 769, 784, 791, 792, 794, 799, 811,
#  813, 817, 826, 830}
# miss_center = set(range(456, 624))
# bad = {64, 68, 123, 139, 172, 175, 181, 217, 287, 340, 422, 428, 444, 447, 454, 460, 463, 464, 
#        470, 473, 475, 486, 487, 488, 498, 630, 637, 694, 771, 804, 861}
# A = set(range(832)) - (B | bad | miss_center)
# 
# def get_contours(x):
#     _, cnts, _ = cv2.findContours(x.astype('uint8'),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
#     return cnts
