#!/usr/bin/env python
# coding: utf-8


# Import all packages
print("[INFO]: importing packeges")
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import sys


# chdir. it depends on the platform
if sys.platform == 'linux':
    # if os is linux cd to
    project_path = "/home/mate/GeFace/"
    NUMBER_OF_DATA = 100000
elif sys.platform is 'windows':
    pass
else:
    pass

os.chdir(project_path)


# In[3]:


try:
    print(os.getcwd())
    # Open CSV with all informations
    csv_file = pd.read_csv("faces_colored/faces_correct.csv",delimiter = ',', encoding = "ISO-8859-1", engine='python')
    pd.set_option('display.max_columns', 100)
except (FileNotFoundError):
    print("CSV file not found")
    current_path = os.getcwd()
    print("Current path is " + current_path)


# In[4]:


csv_file.head()


# In[5]:


#df = csv_file.drop(columns=["nr"])
df = csv_file 


# In[6]:


# convert 1.0 to m as male
#         0.0 to f as female
def mod(x):
    if x == 1.0:
        x = "m"
    else:
        x = "f"
    return x

df["gender"] = df["gender"].apply(mod)


# In[7]:


df.head()


# In[8]:


# create dataset for testing the network
import random
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


#proba_df = df.head(NUMBER_OF_DATA)
proba_df = df

# Randomize but always the same random numbers
np.random.seed(42)
random.seed(42)
# shuffle rows
proba_df = shuffle(proba_df)


# In[9]:


proba_df.head()


# In[10]:


# calculate test train valid data numbers
test_num = int(np.floor(0.1 * proba_df.shape[0]))
valid_num = int(np.floor(0.2 * proba_df.shape[0]))
train_num = int(proba_df.shape[0] - test_num - valid_num)
print("train: {} | valid: {} | test: {}".format(train_num, valid_num, test_num))


# In[11]:


# split the data into train valid and test data
train_data = proba_df.iloc[0:train_num, :]

valid_data = proba_df.iloc[train_num:train_num + valid_num, :]

test_data = proba_df.iloc[ train_num+valid_num:, :]

print("train: {} | valid: {} | test: {}".format(train_data.shape, valid_data.shape, test_data.shape))


# In[12]:


image_path = "faces_colored/"
x_train_p = image_path + train_data['full_path'].values
x_valid_p = image_path + valid_data['full_path'].values
x_test_p = image_path + test_data['full_path'].values

# x_train_l = image_path + train_data['age'].values
# x_valid_l = image_path + train_data['age'].values
# x_test_l = image_path + train_data['age'].values

#x_test_p.shape
#x_test_p[0]


# In[13]:


# get the ages
y_train_age = train_data['age'].values
y_valid_age = valid_data['age'].values
y_test_age = test_data['age'].values

print("age:" ,len(y_train_age), " | ", len(y_valid_age))
y_train_age


# In[14]:


y_train_gender = train_data['gender'].values
y_valid_gender = valid_data['gender'].values
y_test_gender = test_data['gender'].values
print("gender:", len(y_train_gender), " | ", len(y_valid_gender))

# convert to string
y_train_age = y_train_age.astype("str")
y_valid_age = y_valid_age.astype("str")
y_test_age = y_test_age.astype("str")



proba_df.head()




# import the necessary packages
import sys
import os
import PIL
# import the necessary packages
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tfÍ
from keras.optimizers import *
from keras.applications import Xception
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.utils.vis_utils import plot_model

 
# import the necessary packages
#from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
# from pyimagesearch.smallervggnet import SmallerVGGNet
# from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os


class FaceNet:
    
    @staticmethod
    def build_ages_branch(inputs, numAges, finalAct="softmax", chanDim=-1):
        # utilize a lambda layer to convert the 3 channel input to a
        # grayscale representation
        x = Conv2D(32, (3, 3), padding="same")(inputs) 
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        predictions = Dense(numAges, activation=finalAct,name="ages_output")(x)
        # add your top layer block to your base model

        #print(model.summary())
    
        # define a branch of output layers for the number of different
        # ages
        # return the category prediction sub-network
        return predictions
    

    
    @staticmethod
    def build_gender_branch(inputs, numGender, finalAct="softmax", chanDim=-1):
        
        padding = "same"
        # CONV => RELU => POOL
        x = Conv2D(32, (3, 3), padding=padding)(inputs) 
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)
        
        # CONV => RELU => POOL
        x = Conv2D(32, (3, 3), padding=padding)(x) 
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)
        
        # (CONV => RELU) * 2 => POOL
        x = Conv2D(64, (5, 5), padding=padding)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Conv2D(64, (3, 3), padding=padding)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        
        
        # (CONV => RELU) * 2 => POOL
        x = Conv2D(64, (5, 5), padding=padding)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Conv2D(64, (3, 3), padding=padding)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
 
        # (CONV => RELU) * 2 => POOL
        x = Conv2D(128, (3, 3), padding=padding)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Conv2D(128, (3, 3), padding=padding)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        
        # define a branch of output layers for the number of different
        # genders
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(numGender)(x)
        x = Activation(finalAct, name="gender_output")(x)
 
        # return the color prediction sub-network
        return x
    
    @staticmethod
    def build(width, height, numAges, numGenders, finalAct="softmax"):
        # initialize the input shape and channel dimension (this code
        # assumes you are using TensorFlow which utilizes channels
        # last ordering)
        inputShape = (height, width, 3)
        chanDim = -1
        #base_model = Xception(input_shape=inputShape, 
#                               weights=None, include_top=False)
        #f#or layer in base_model.layers:
         #   layer.trainable = 
        # construct both the "category" and "color" sub-networks
        inputs = Input(shape=inputShape)
        
        agesBranch = FaceNet.build_ages_branch(inputs,
            numAges, finalAct=finalAct, chanDim=chanDim)
        
        genderBranch = FaceNet.build_gender_branch(inputs,
            numGenders, finalAct=finalAct, chanDim=chanDim)
 
        # create the model using our input (the batch of images) and
        # two separate outputs -- one for the clothing category
        # branch and another for the color branch, respectively
        model = Model(
            inputs=inputs,
            outputs=[agesBranch, genderBranch],
            )
        
        print(model.summary())
        plot_model(model, show_shapes=True, show_layer_names=True)
        # return the constructed network architecture
        return model


# In[20]:


# fix seed for reproducible results (only works on CPU, not GPU)
seed = 42
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 100
INIT_LR = 1e-3
BS = 64
IMAGE_DIMS = (96, 96, 3)



lb_gender = LabelBinarizer()
lb_age = LabelBinarizer()

y_train_gender = lb_gender.fit_transform(y_train_gender)
y_train_age = lb_age.fit_transform(y_train_age)
y_train_gender = [y_train_gender, 1-y_train_gender]

y_valid_gender = lb_gender.transform(y_valid_gender)
y_valid_age = lb_age.transform(y_valid_age)
y_valid_gender = [y_valid_gender, 1-y_valid_gender]


# In[23]:


project_path+x_valid_p[0]


# In[24]:


np.array(y_train_age)[1:41].shape


# In[25]:


# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest",
#     brightness_range=(0.2,1.0),
                        )


# In[26]:


def trainImageLoader(files_x, y_train_age, y_train_gender, batch_size,L):

    #this line is just to make the generator infinite, keras needs that    
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            train_x = []
            for file in files_x[batch_start:limit]:
                image = cv2.imread(project_path+file)
                image = img_to_array(image)
                train_x.append(image)
            train_x = np.array(train_x, dtype="float32") / 255.0
            train_y = y_train_age
            
            y_train_g = np.array(y_train_gender).squeeze().T[batch_start:limit]
            
            y_train_a = np.array(train_y)[batch_start:limit]
        

            yield (train_x,{"ages_output": y_train_a, "gender_output": y_train_g}) #a tuple with two numpy arrays with batch_size samples     

            batch_start += batch_size   
            batch_end += batch_size


# In[27]:


def validImageLoader(files_x, y_valid_age, y_valid_gender, batch_size,L):

    #this line is just to make the generator infinite, keras needs that    
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            valid_x = []
            for file in files_x[batch_start:limit]:
                image = cv2.imread(project_path+file)
                image = img_to_array(image)
                valid_x.append(image)
            valid_x = np.array(valid_x, dtype="float32") / 255.0
            valid_y = y_valid_age
            
            y_valid_g = np.array(y_valid_gender).squeeze().T[batch_start:limit]
            y_valid_a = np.array(valid_y)[batch_start:limit]
        

            yield (valid_x,{"ages_output": y_valid_a, "gender_output": y_valid_g}) #a tuple with two numpy arrays with batch_size samples     

            batch_start += batch_size   
            batch_end += batch_size


# In[28]:


# initialize the model
print("Ages number", len(lb_age.classes_))
print("[INFO] compiling model...")
# load model
model = FaceNet.build(IMAGE_DIMS[0], IMAGE_DIMS[1], numAges=len(lb_age.classes_), 
                      numGenders=2,finalAct="softmax")
# create optimazitions method
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

#model.compile(loss="categorical_crossentropy", optimizer=opt,
#	metrics=["accuracy"])
losses = {
	"ages_output": "categorical_crossentropy",
    "gender_output": "categorical_crossentropy"
}
lossWeights = {"ages_output": 1.0, "gender_output": 1.0}

model.compile(loss=losses, optimizer=opt,
	metrics=["accuracy"])

# Create callback list for checkpoint and Earlystopping
callbacks_list = [
        ModelCheckpoint(project_path+"gen_ages_model.hdf5", monitor='val_acc', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=40, verbose=1)
    ]

# train the network
print("[INFO] training network...")
TBS = 64

History = model.fit_generator(
 	trainImageLoader(x_train_p,y_train_age, y_train_gender,TBS,len(x_train_p)),
    steps_per_epoch=len(x_train_p) // TBS,
	validation_data=validImageLoader(x_valid_p,y_valid_age,y_valid_gender,TBS,len(x_valid_p)),
    validation_steps = len(x_valid_p) // TBS,
    epochs=EPOCHS,
    verbose=1,
    callbacks=callbacks_list,
	
)

# save the model to disk
print("[INFO] serializing network...")
model.save(project_path+"gen_ages_model.hdf5")
 
# save the category binarizer to disk
print("[INFO] serializing category label binarizer...")
f = open(project_path+"gen_ages.label", "wb")
f.write(pickle.dumps(lb_age))
f.close()
 


# In[ ]:


REAL_EPOCH = len(History.epoch)
# print(y_train_gender)


# In[ ]:



len(History.epoch)


# In[ ]:


loss_names = ["val_loss"]
plt.style.use("ggplot")

(fig, ax) = plt.subplots(2, 1, figsize=(8,8))

for (i, l) in enumerate(loss_names):
    ax[i].set_title("Loss for {}".format(l))
    ax[i].set_xlabel("Epochs")
    ax[i].set_ylabel("Loss")
    ax[i].plot(np.arange(0, REAL_EPOCH), History.history[l], label=l)
    ax[i].plot(np.arange(0, REAL_EPOCH), History.history[l],label=l)
    ax[i].legend()
    
    plt.tight_layout()
    plt.savefig("multi_loss.png")


# In[ ]:


accuracy_names = ["val_acc"]
plt.style.use("ggplot")

(fig, ax) = plt.subplots(2, 1, figsize=(8,8))

for (i, l) in enumerate(accuracy_names):
    ax[i].set_title("Accuracy for {}".format(l))
    ax[i].set_xlabel("Epochs")
    ax[i].set_ylabel("Accuracy")
    ax[i].plot(np.arange(0, REAL_EPOCH), History.history[l], label=l)
    ax[i].plot(np.arange(0, REAL_EPOCH), History.history[l],label=l)
    ax[i].legend()
    
    plt.tight_layout()
    plt.savefig("multi.png")
    


# In[ ]:


# create a new figure for the accuracies
accuracyNames = ["acc"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(2, 1, figsize=(8, 8))
 
# loop over the accuracy names
for (i, l) in enumerate(accuracyNames):
	# plot the loss for both the training and validation data
	ax[i].set_title("Accuracy for {}".format(l))
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Accuracy")
	ax[i].plot(np.arange(0, REAL_EPOCH), History.history[l], label=l)
	ax[i].plot(np.arange(0, REAL_EPOCH), History.history["val_" + l],
		label="val_" + l)
	ax[i].legend()
 
# save the accuracies figure
plt.tight_layout()
plt.savefig("multi_accs.png")


# In[ ]:


from keras.models import load_model


# In[ ]:



# load the image
image = cv2.imread(project_path+"p_barbi.jpg")
#image = cv2.imread(project_path+"faces_colored/01/nm0000001_rm124825600_1899-5-10_1968.jpg")
# image = cv2.imread("/home/mate/Pictures/dorka1.jpg")
face_cascade = cv2.CascadeClassifier('detector/haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

bb = []
i = 0
output = image.copy()
for (x,y,w,h) in faces:
    bb.append([w * h, i])
    cv2.rectangle(output,(x,y),(x+w,y+h),(255,0,0),2)
 #   roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    i += 1
    print(i)
    break

# pre-process the image for classification
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network from disk, followed
# by the category and color label binarizers, respectively
print("[INFO] loading network...")
model = load_model("gen_ages_model.hdf5", custom_objects={"tf": tf})
agesLB = pickle.loads(open("gen_ages.label", "rb").read())
 
# classify the input image using Keras' multi-output functionality
print("[INFO] classifying image...")
(agesProba ) = model.predict(image)
 
# find indexes of both the category and color outputs with the
# largest probabilities, then determine the corresponding class
# labels
agesIdx = agesProba[0].argmax()

agesLabel = agesLB.classes_[agesIdx]

# draw the category label and color label on the image
agesText = "age: {} ({:.2f}%)".format(agesLabel,
	agesProba[0][agesIdx] * 100)


cv2.putText(output, agesText, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# display the predictions to the terminal as well
print("[INFO] {}".format(agesText))
# show the probabilities for each of the individual labels
output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
# show the output image
# print("[INFO] {} years old".format(label))
plt.imshow(output,)


# In[ ]:


History.model.count_params()


# In[ ]:




