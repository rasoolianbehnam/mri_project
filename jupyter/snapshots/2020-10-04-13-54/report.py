#!/usr/bin/env python
# coding: utf-8

# In[46]:


# imports
# %reset -f
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from pylab import rcParams
rcParams['figure.figsize'] = 8, 8
from mri_project.custom_imports import *
from mri_project.pipeline import  predict_image
from mri_project.utility import MriImage
import math


# # Loading stuff

# In[2]:


m11 = tf.keras.models.load_model(
    "../data/models/new_era.v03.11muscles.h5"
)
m9  = tf.keras.models.load_model(
    "../data/models/new_era.v02.9muscles.h5"
)


# In[3]:


data_dir = "../data/data/images_with_predictions_v07/"


# In[4]:


all_data_files = glob.glob(f"{data_dir}/*data", recursive=True)
assert len(all_data_files)  == 832


# # Images

# ## Sample raw/traced images

# In[45]:


def show_files(files):
    fig, axes = plt.subplots(len(files), 2, figsize=(8*1.5, 10*1.5))
    fig.tight_layout(pad=0, h_pad=0.5, w_pad=0 )
    for i, file in enumerate(files):
        data = joblib.load(file)
        axes[i, 0].imshow(data.raw_image)
        axes[i, 1].imshow(data.traced_image)
    for ax in axes.reshape(-1):
        ax.axis('off')
        
        
files_ = [
    '../data/data/images_with_predictions_v07/3-WF2B-11_16PostRaw.data',
    '../data/data/images_with_predictions_v07/4-WM9B-15_16RepPostRaw.data',
    '../data/data/images_with_predictions_v07/1-COF5-5_50ConRaw.data'
]
show_files(files_)


# ## Traced lever arms

# In[49]:


def show_traced_lever_arms(files):
    fig, axes = plt.subplots(len(files), 2, figsize=(8*1.5, 10*1.5))
    fig.tight_layout(pad=0, h_pad=0.5, w_pad=0 )
    for i, file in enumerate(files):
        data = joblib.load(file)
        data.get_traced_contours(90)
        axes[i, 0].imshow(data.raw_image)
        axes[i, 1].imshow(data.traced_lever_arm_images[90])
    for ax in axes.reshape(-1):
        ax.axis('off')

show_traced_lever_arms(files_)


# ## An example of Original, traced, and predicted image

# In[16]:


def demonstrate_file(file):
    if isinstance(file, str):
        data = joblib.load(file)
    else:
        data = file
    n_images = 4
       
    images = [data.raw_image, data.traced_image, data.traced_binary_mask]
    titles = ['Raw Image', 'Traced Image', 'Masked Traced Image']
    if 'traced_multilabel_mask' in data.get_attributes():
        images.append(data.traced_multilabel_mask)
        titles.append('Labeled Traced Image')
    images.append(data.predicted)
    titles.append('Predicted Image')
    if 'predicted_postprocessed' in data_.get_attributes():
        images.append(data.predicted_postprocessed)
        titles.append('Post-processed Predictions')
        
    n_cols = 2
    n_rows = math.ceil(len(images) / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*1.5, 10*1.5))
    fig.tight_layout(pad=0, h_pad=0.5, w_pad=0 )
    axes = axes.reshape(-1)
    
    for image, title, ax in zip(images, titles, axes):
        ax.imshow(image)
        ax.set_title(title)
        
    for ax in axes:
        ax.axis('off')
        
    for i in range(len(images), len(axes)):
        axes[i].axis('off')


# In[38]:


#alternatives
# file = np.random.choice(all_data_files)
file = '../data/data/images_with_predictions_v07/3-WF2B-11_16PostRaw.data'
# file = '../data/data/images_with_predictions_v07/4-WM9B-15_16RepPostRaw.data'
# file = '../data/data/images_with_predictions_v07/1-COF5-5_50ConRaw.data'
print(file)
data_ = joblib.load(file)
demonstrate_file(data_)


# ## Comparing original lever arms with predicted lever arms

# In[39]:


# Drawing traced and predicted lever arms
data_ = joblib.load(file)
angle = 90
data_.get_traced_contours(90)
data_.predicted = data_.predicted_postprocessed
data_.get_predicted_contours(angle)
fig, axes = plt.subplots(1, 2, figsize=(15, 15))

axes[0].imshow(data_.traced_lever_arm_images[90])
axes[0].set_title("Traced Image Lever Arms")
axes[1].imshow(data_.predicted_lever_arm_images[90])
axes[1].set_title("Predicted Image Lever Arms")
for ax in axes:
    ax.axis('off')


# In[ ]:




