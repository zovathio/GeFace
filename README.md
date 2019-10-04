# -GeFace
Student project for Deep Learning. Forked from @bartfaimate

Participants:
<br>Máté Bártfai (@bartfaimate)
<br>Örkény Zováthi (@zovathio)

The documentation of the project (idea, theory, results) can be found [here](https://github.com/bartfaimate/GeFace/blob/master/GeFace_paper.pdf).

## Requirements
Python 3.5 (at least)<br>
Tensorflow, Keras<br>
NumPy, Pandas, sklearn<br><br>

You can download this packages with `pip install <packagename>` command

--------------------------------------------------------------------------------------------

## Project structure

## faces_colored folder

This folder contains the metadata as a csv file. (Image paths, gender, age)
Images can be found [here](https://drive.google.com/open?id=14I8YEH0egjkkbrpaBIAcQuoM9cGHVUoH)



## notebooks folder


**It could be, that you have to change the paths in the whole project**
In the notebooks you can find more descriptions about the processes.

The notebooks folder contains the jupyter notebooks. 
For the preprocessing the database you have to run **preprocess_step\[0:2\].ipynb** notebooks.

For training you have to run **Xception-regression-finetune_class.ipynb**. This notebook contains the database processing, model building, compile and training process.

With the **plotter.ipynb** notebook you can plot and save the training curves.

**Xception-regression_classify.ipynb** notebook loads the model, labels and do the prediction on an image 

**Xception-regression_test.ipynb** loads the model and do the prediciton on the test database, which contains around 26 000 pictures and saves the results into a csv file

**prev_approaches** folder contains prevoius approaches. You don't have to look at them.


## src folder
**It could be, that you have to change the paths in the whole project**

This folder contains the runnable python script. In the **titan** folder it can be found which script we have ran on a TitanX GPU.

**Xception-regression_finetun.py** is generated from the notebook file, which has the same name.


## Summary

The neccessary files are: notebooks/preprocess_step\[0:2\].ipynb, notebooks/Xception-regression-finetune_class.ipynb, notebooks/Xception-regression_test.ipynb, notebooks/Xception-regression_classify.ipynb

The files of the trained and tested model can be found [here](https://drive.google.com/open?id=1NVvmUBpyR1OPnH60hpmrJhwDWaTz7PPG)

-------------------------------------------------------------------------------------------------------------------


## Declaration of goal
Our goal is to construct an intelligent system which is able to predict the age (and gender) of people from a single image using Convulutional Neural Networks (CNNs). We are also interested deeply in the egsistence of any visible features which can represent the aging on the face - like the growth rings on trees for example.

## Step 1: Introduction, literature overview
Age estimation is still an open and unsolved task in today's life. Althoguh, in the past few years a lot of different approaches were created and presented. Some of these methods collect a lot of information about the person (e.g. height, weight, favourites, family status -- see [this What-If-Tool demo](https://pair-code.github.io/what-if-tool/age.html)), other approaches look at a photo of the whole body and make some consequences. 

Our proposal is that without any additional information or without looking at the whole body structure, the correct age can be predicted from a single image of the face. The face is an individual and specific attribute and with the help of Convolutional Neural Networks, we want to learn its features and use it for correct predictions. 

Our other goal is to look into the deeper layers of the Network to see which features dominate in representing the aging on the face, like such networks which learnes styles and use it for transferring images into different styles and ages. (For an easy example see  [this article](https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398).)

We are currently not aware of the possible outcome of our network. In case of good accuracy, our system can be used on its own to predict ages, in case of a worse accuracy, it can be still combined with the other techniques that were mentioned above. But in both cases, the learned features can be visualized and checked, which is - as we already said - also a goal of the project.

## Step 2: Dataset and preprocessing
In the project we downloaded a dataset with ca. 500 000 images and annotations. Then we formed, cropped and resized it with the help of the built-in OPENCV face detection tool into 96x96 pixel images that contains only the faces of people. (For Milestone 1: The formed dataset (~1,5 GB) is available [here](https://drive.google.com/open?id=14I8YEH0egjkkbrpaBIAcQuoM9cGHVUoH) for research purpose.)
<br> The three steps we made are the following:<br>
Step 0: Cropping the dataset and remove all invalid items, visualize the numbers of each class.<br>
Step 1: Resize all the images.<br>
Step 2: Split the dataset to train-validation-test-values. Result is x_train, x_valid, x_test input images and y_train, y_valid, y_test output values.

## Step 3: Model constuction, training, validation and test
We created a model architecture based on VGG8. Input dimensions of the network are 96*96@3. In our very first training process (for Milestone 2) we built, trained and tested the model in the training.ipynb notebook.
The steps are the following:
* We randomly selected 1000 pictures. 700 for training 200 for validation and 100 for testing. 
* EarlyStopping and Checkpoint is not implemented yet
* Training accuracy and validation accuracy could be better -> we will modify the model till the presetation
* We tested on 2 selfies and the DNN showed us 27 years old but we are 23 and 24 so it is not so bad result.
* Till the end of semester we will feed up the network with all the pictures in the dataset (~260k colored pictures)

## Step 4: Deployment
Without limitation, a few problems that can be solved with correct age estimation. These were also our motivation for choosing such a difficult and unsolved task.

* Determine the age of people with undocumented birth.
* Measuring the avarage age of audience in presentations, advertisements, etc..
* Determine the feautures which represent tha aging on the face (like e.g. pleats)
* More examples in progress...
