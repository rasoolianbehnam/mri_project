import cv2
import numpy as np
import joblib
from mri_project.pipeline import predict_image
from mri_project.utility import get_muscles
from mri_project.utility import draw_lever_arms

from mri_project.contour_ops import get_muscle_contours, sort_muscle_contours_by_dist_from_center, get_muscle_contours_dict
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)




def read_image(x):
    if isinstance(x, str):
        x = cv2.imread(x)
    elif not isinstance(x, np.ndarray):
        raise ValueError("x should be either a path to image or an image")
    return x

def read_generic(x):
    if isinstance(x, str):
        x = joblib.load(x)
    return x

def show_lever_arms(img, angle, numbered=False, scale=1, 
                    ax=None, plot=True, img_color_coefficient=1):
    if angle > np.pi:
        angle = np.pi / 180 * angle
    if numbered:
        good_cnts = get_muscle_contours_dict(img)
        if good_cnts.get(0) is not None:
            del good_cnts[0]
        # print(good_cnts.keys())
        sorted_cnts = [good_cnts[i][0] for i in sorted(good_cnts.keys()) if len(good_cnts[i])]
    else:
        good_cnts = get_muscle_contours(img)
        sorted_cnts = sort_muscle_contours_by_dist_from_center(good_cnts)
    if len(good_cnts) not in {9, 11}:
        logger.warning("muscles not of size 9 or 11")
        if len(good_cnts) > 11:
            sorted_cnts = sorted_cnts[:11]
    logger.info(f"Number of muscles = {len(sorted_cnts)}")
    center_point = np.int32(np.mean(sorted_cnts[0], axis=(0, 1))).reshape(-1)
    out, lever_arms = draw_lever_arms(img, sorted_cnts, angle, center_point, scale)
    out = img_color_coefficient * img+out
    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.imshow(out)
    return sorted_cnts, lever_arms, out

class MuscleDetector(object):
    traced_lever_arm_images = {}
    predicted_lever_arm_images = {}
    traced_features = {}
    predicted_features = {}
    traced_binary_mask = None
    traced_multilabel_mask = None
    predicted = None
    traced_contours = None
    predicted_contours = None
    
    def __init__(self, id_, raw_image, scale, traced_image=None):
        self.id = id_
        self.raw_image = read_image(raw_image)
        self.scale = scale
        self.traced_image = read_image(traced_image) if traced_image is not None else None
        
    def get_traced_binary_mask(self, img=None):
        if img is None:
            img = self.traced_image
        out = get_muscles(img)
        self.traced_binary_mask = out
        return out
    
    def get_traced_multilabel_mask(self, unmatched_pixel_value=0):
        rszd = cv2.resize(self.traced_binary_mask, self.predicted.shape[::-1])
        cnts = get_muscle_contours(rszd)
        cnt_map = {}
        predicted_unique = np.unique(self.predicted)
        result = np.zeros_like(rszd)
        for i, cnt in enumerate(cnts, 1):
            max_overlap = .1
            chosen_j = unmatched_pixel_value
            for j in predicted_unique:
                if j == 0: continue
                imt = np.zeros_like(rszd)
                cv2.drawContours(imt, [cnt], -1, 1, -1)
                common = (imt > 0) & (self.predicted > 0) & ((imt>0) == (self.predicted == j))
                overlap = np.sum(common)
                if overlap > max_overlap:
                    max_overlap = overlap
                    chosen_j = j
                # plt.imshow(common)
                # plt.show()
            cnt_map[i] = chosen_j
            cv2.drawContours(result, [cnt], -1, int(chosen_j), -1)
        if len(cnt_map.keys()) != len(set(cnt_map.values())):
            logger.warning(f"Matches not 100%. The map is {cnt_map}")
        self.traced_multilabel_mask = result
        return result
    
    def has_good_prediction(self, refresh=False):
        if refresh:
            self.get_traced_multilabel_mask()
        return len(np.unique(self.predicted)) == len(np.unique(self.traced_multilabel_mask))
        
    
    def get_contour_areas(self):
        self.traced_features['area']    = [cv2.contourArea(x)*(self.scale**2) for x in self.traced_contours]
        self.predicted_features['area'] = [cv2.contourArea(x)*(self.scale**2) for x in self.predicted_contours]
        
    def get_contour_centers(self):
        self.traced_features['center']    = [x.mean(axis=(0,1)) for x in self.traced_contours]
        self.predicted_features['center'] = [x.mean(axis=(0,1)) for x in self.predicted_contours]
 
        
    def predict(self, model):
        self.predicted = np.uint8(predict_image(model, self.raw_image))
    
    def get_traced_contours(self, angle, img_color_coefficient=1/11):
        if self.traced_image is None:
            return
        if self.has_good_prediction():
            cnts, cnt_features, lever_image = show_lever_arms(self.traced_multilabel_mask, angle, True, self.scale, 
                                                              plot=False, img_color_coefficient=img_color_coefficient)
        else:
            im_floodfill = self.get_traced_binary_mask(self.traced_image)
            cnts, cnt_features, lever_image = show_lever_arms(self.traced_binary_mask, angle, False, self.scale, plot=False)
        self.traced_contours = cnts
        self.traced_lever_arm_images[angle] = lever_image
        self.traced_features[f'lever_arm_{angle}'] = [x['lever_arm'] for x in cnt_features]
        
    def get_predicted_contours(self, angle, img_color_coefficient=1/11):
        cnts, cnt_features, lever_image = show_lever_arms(self.predicted, angle, True, self.scale, 
                                                     plot=False, 
                                                     img_color_coefficient=img_color_coefficient)
        self.predicted_contours = cnts
        self.predicted_lever_arm_images[angle] = lever_image
        self.predicted_features[f'lever_arm_{angle}'] = [x['lever_arm'] for x in cnt_features]
    
    def get_attributes(self):
        return [x for x in dir(self) if not x.startswith('_') and not callable(getattr(self, x))]
    
    @classmethod
    def load_from_dict(cls, d):
        x = cls(d['id'], d['raw_image'], d['scale'])
        attrs = set(x.get_attributes())
        assert len(d.keys() - attrs) == 0
        for k, v in d.items():
            setattr(x, k, v)
        return x
    
    def save_to_dict(self):
        out = {}
        attrs = self.get_attributes()
        for attr in attrs:
            out[attr] = getattr(self, attr)
        return out