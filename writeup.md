# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./docs/netmodel.png "Model Visualization"
[image2]: ./docs/center_image.jpg "Center Image"
[image3]: ./docs/left.jpg "Recovery Image"
[image4]: ./docs/right.jpg "Recovery Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolutional neural network inspired by nvidia's network. A sequence of 5 convolutional layers followed by 4 dense layers. All in file *dnn_nvidia_steering.py*.

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer near the end in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 17). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 33).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
I used a combination of center lane driving, with side cameras for recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to experiment with a simple network with proven efficiency on relate task.

My first step was to use transfer learning on InceptionV3 and unfreezing the top to inception modules, but it was taking too long to train and I realised that it was an overkill for the task in hand.
Thus, I choose a convolution neural network model similar to Nvidia's as it was tested on similar conditions.

To prevent overfitting and help generalization, I modified the model to use pooling instead of strides, and a final dropout layer. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, specially on steep curves, to improve the driving behavior, I generated more data of similar cases. But I also realized that some instability was introduced because of the simulation time, to solve that, I simply had to reduce the rendering quality.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 20-26 and file *dnn_nvidia_steering.py*) consisted of a convolution neural network with the following layers and layer sizes:

|Layer (type)         |        Output Shape       |       Param #|   
|:---------------------:|:-------------------------:|:--------------------:|
|cropping2d_1 (Cropping2D)  |  (None, 65, 320, 3)  |      0|         
|lambda_1 (Lambda)            |(None, 65, 320, 3)   |     0         |
|conv2d_1 (Conv2D)            |(None, 61, 316, 24)   |    1824      |
|max_pooling2d_1 (MaxPooling2D) |(None, 30, 158, 24)    |   0         |
|conv2d_2 (Conv2D)            |(None, 26, 154, 36)    |   21636     |
|max_pooling2d_2 (MaxPooling2D) |(None, 13, 77, 36)      |  0         |
|conv2d_3 (Conv2D)            |(None, 9, 73, 48)        | 43248     |
|conv2d_4 (Conv2D)            |(None, 7, 71, 64)        | 27712     |
|conv2d_5 (Conv2D)            |(None, 5, 69, 64)        | 36928     |
|flatten_1 (Flatten)          |(None, 22080)             |0         |
|dense_1 (Dense)              |(None, 100)               |2208100   |
|dense_2 (Dense)              |(None, 50)     |           5050      |
|dense_3 (Dense)              |(None, 10)      |          510       |
|dropout_1 (Dropout)          |(None, 10)       |         0         |
|dense_4 (Dense)              |(None, 1)         |        11        |
Total params: 2,345,019
_________________________________________________________________

Here is a visualization of the architecture

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving in one direction, than another two laps in the other direction. Here is an example image of center lane driving:

![alt text][image2]

I also used the left side and right sides images to show what a recovery could look like:

![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help to prevent the model to have bias toward one side.


After the collection process and data augmentation, I had more than 25 thousand data points. I then preprocessed this data by mean centering and scaling.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting, both loss and val_loss were decreasing without sign of overfitting. 
The ideal number of epochs was 15 as evidenced by the valley on loss/val_loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
