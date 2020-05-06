from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, MaxPooling2D, Dropout


def make_nvidia_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Convolution2D(24, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(36, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(48, 5, 5, activation='relu'))
    # model.add(MaxPooling2D())
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dropout(0.8))
    model.add(Dense(1))
    return model
