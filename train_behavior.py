import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import csv
import glob
samples = []

for log_file in glob.glob('../SimulatorData/run*/driving_log.csv'):
    with open(log_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
        
print('Size of samples is {}'.format(len(samples)))

def preprocess(image):
    lab_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab_image[:,:,0] = cv2.equalizeHist(lab_image[:,:,0])
    return (lab_image-127.5)/127.5

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from pathlib import Path
from keras.layers import Input, Lambda, Cropping2D
import tensorflow as tf

here = Path('/home/eelael/courses/SelfDriving/BehaviourClonning/')

from sklearn.model_selection import train_test_split
from keras.applications.inception_v3 import preprocess_input
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import sklearn
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []            
            
            for batch_sample in batch_samples:
                if len(batch_sample) == 0:
                    continue                    
                
                center_image = mpimg.imread( '..' / Path(batch_sample[0]).relative_to(here)).astype(np.float32)
                preprocess_input(center_image)
                left_image = mpimg.imread( '..' / Path(batch_sample[1]).relative_to(here)).astype(np.float32)
                preprocess_input(left_image)
                right_image = mpimg.imread( '..' / Path(batch_sample[2]).relative_to(here)).astype(np.float32)
                preprocess_input(right_image)
                
                steering_center = float(batch_sample[3])
                
                # create adjusted steering measurements for the side camera images
                correction = 0.1 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                
                images.extend((left_image, right_image, center_image))
                angles.extend((steering_left, steering_right, steering_center))
                
                # flipped image
                images.append(np.fliplr(center_image))
                angles.append(-steering_center)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set a couple flags for training - you can ignore these for now
weights_flag = 'imagenet' # 'imagenet' or None

# Loads in InceptionV3
from keras.applications.inception_v3 import InceptionV3


# Using Inception with ImageNet pre-trained weights
# which will speed up training. Keras v2.0.9 supports down to 139x139x3
inception = InceptionV3(weights=weights_flag, include_top=False,
                        input_shape=(139,139,3))

for layer in inception.layers[:249]:
       layer.trainable = False
for layer in inception.layers[249:]:
       layer.trainable = True


# Makes the input placeholder layer 32x32x3 for CIFAR-10
cifar_input = Input(shape=(160,320,3))
cropped_input = Cropping2D(cropping=((70,25), (0,0)))(cifar_input)

# Re-sizes the input with Kera's Lambda layer & attach to cifar_input
# which will speed up training. Keras v2.0.9 supports down to 139x139x3
resized_input = Lambda(lambda image: tf.image.resize_images( 
    image, (139, 139)))(cropped_input)

# Feeds the re-sized input into Inception model
# You will need to update the model name if you changed it earlier!
tf.get_variable_scope().reuse_variables()
inp = inception(resized_input)

# Imports fully-connected "Dense" layers & Global Average Pooling
from keras.layers import Dense, GlobalAveragePooling2D

## TODO: Setting `include_top` to False earlier also removed the
##       GlobalAveragePooling2D layer, but we still want it.
##       Add it here, and make sure to connect it to the end of Inception
avg_layer = GlobalAveragePooling2D()(inp)

## TODO: Create two new fully-connected layers using the Model API
##       format discussed above. The first layer should use `out`
##       as its input, along with ReLU activation. You can choose
##       how many nodes it has, although 512 or less is a good idea.
##       The second layer should take this first layer as input, and
##       be named "predictions", with Softmax activation and 
##       10 nodes, as we'll be using the CIFAR10 dataset.
out = Dense(512, activation='relu')(avg_layer)
predictions = Dense(1)(out)

from keras.models import Model

model = Model(inputs=cifar_input, outputs=predictions)

# Compile the model
# from keras.optimizers import 
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='mse')

# Use a generator to pre-process our images for ImageNet
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Train the model
from numpy import ceil
batch_size = 64
epochs = 100
# Note: we aren't using callbacks here since we only are using 5 epochs to conserve GPU time
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=epochs)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# from keras.models import load_model
# model = load_model('inception_retrained_model.h5', custom_objects={'tf':tf, 'input_size': 139})
from os.path import isfile
if isfile('inception_retrained_model_v2.h5'):
    from keras.models import load_model
    tf.reset_default_graph()
    model = load_model('inception_retrained_model_v2.h5', custom_objects={'tf':tf })

model.fit_generator(train_generator,
                    steps_per_epoch=ceil(len(train_samples)/ batch_size),
                    validation_data=validation_generator,
                    validation_steps=ceil(len(validation_samples)/batch_size),
                    epochs=epochs, verbose=1)

model.save('inception_retrained_model_v2.h5')