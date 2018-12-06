#!/usr/bin/env python
# coding: utf-8

# In[6]:


from keras.models import load_model
# Import all packages

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import sys
from keras.preprocessing.image import img_to_array
import tensorflow as tf
import pickle
from sklearn.preprocessing import MinMaxScaler

get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


project_path = "/home/mate/develop/PycharmProjects/GeFace/"

# load the trained convolutional neural network from disk, followed
# by the category and color label binarizers, respectively
print("[INFO] loading network...")
model = load_model(project_path  + "faces_model/last/xception_regression_age_fine.hdf5", custom_objects={"tf": tf})
agesLB = pickle.loads(open(project_path + "faces_model/last/regression.label", "rb").read())


# In[8]:


test_csv = "/home/mate/develop/PycharmProjects/GeFace/test/test_data.csv"
df = pd.read_csv(test_csv)


# In[9]:


df.head()


# In[ ]:


# classify the input image using Keras' multi-output functionality
BS = 128
df['preds_age'] = 0
df['preds_gender'] = -1
for j in range(len(df)//BS):
    b_images = []
    for i in range(BS):
        image = cv2.imread(project_path+df['x_test_path'][j*BS+i])
        image = img_to_array(image)
        b_images.append(image)
    b_images = np.asarray(b_images)
    b_images = b_images.astype("float") / 255.0*2.0-1
    (agesProba) = model.predict(b_images)
    agesLabels = agesLB.inverse_transform(agesProba[0])
    genderLabels = agesProba[1]
    for ii in range(BS):
        df['preds_age'][j*BS+ii] = agesLabels[ii]
        df['preds_gender'][j*BS+ii] = np.argmax(genderLabels[ii])


# In[ ]:


df.head(32)


# In[14]:


deviation = np.abs(np.asarray(df["preds_age"]-df["y_test_age"]))
deviation.sum() / len(deviation)


# In[21]:


(agesProba) = model.predict(b_images)
agesLabels = agesLB.inverse_transform(agesProba[0])
genderLabels = agesProba[1]


# In[22]:





# In[23]:


for i in range(BS):
    df['preds_age'][i] = agesLabels[i]
    df['preds_gender'][i] = np.argmax(genderLabels[i])


# In[25]:


df.head(20)


# In[20]:





# In[48]:


if np.argmax(genderLabel) == 1:
    gendertext = "female"
else:
    gendertext = "male"


# In[ ]:





# In[49]:


# draw the category label and color label on the image
agesText = "age: ({:.0f})".format(float(agesLabel))
print(gendertext)
fontsize = font_size_calculator(image.shape)
fontwidth = font_width_calculator(image.shape)
print(fontsize)
x_pos, y_pos = calculate_text_position(image.shape)
print(x_pos, y_pos)
cv2.putText(output, agesText, (x_pos, y_pos+10), cv2.FONT_HERSHEY_SIMPLEX,
	fontsize, (255, 0, 0), fontwidth)
cv2.putText(output, gendertext, (x_pos, y_pos*2+20), cv2.FONT_HERSHEY_SIMPLEX,
	fontsize, (255, 155, 0), fontwidth)
# display the predictions to the terminal as well
print("[INFO] {}".format(agesText))
# show the probabilities for each of the individual labels
cv2.imwrite(image_name + "_class.jpg",output)
outp = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
# show the output image
# print("[INFO] {} years old".format(label))
plt.imshow(outp,)


# In[26]:


output.shape


# In[ ]:





# In[ ]:




