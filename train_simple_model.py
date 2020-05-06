from math import ceil
from os.path import isfile

from sklearn.model_selection import train_test_split

from dnn_nvidia_stearing import make_nvidia_model
from helper import get_samples, sample_generator

import tensorflow as tf

# Parameters
simple_model_file = 'nvidia_model.h5'
batch_size = 32
epochs = 10

# load samples
train_samples, validation_samples = train_test_split(get_samples('../SimulatorData'), test_size=0.2)
print('{} train samples and {} validation samples'.format(len(train_samples), len(validation_samples)))

# load or create model
if not isfile(simple_model_file):
    model = make_nvidia_model()
else:
    from keras.models import load_model

    model = load_model(simple_model_file, custom_objects={'tf': tf})

# compile and train the model using the generator function
train_generator = sample_generator(train_samples, batch_size=batch_size)
validation_generator = sample_generator(validation_samples, batch_size=batch_size)

# from keras.optimizers import SGD
# SGD(lr=0.0001, momentum=0.9)
model.compile(optimizer='adam', loss='mse')

model.fit_generator(train_generator,
                    steps_per_epoch=ceil(len(train_samples) / batch_size),
                    validation_data=validation_generator,
                    validation_steps=ceil(len(validation_samples) / batch_size),
                    epochs=epochs, verbose=1)

model.save(simple_model_file)
