from pathlib import Path

import matplotlib.image as mpimg
import csv
import glob

import numpy as np
from sklearn.utils import shuffle


def get_samples(main_dir):
    samples = []
    for log_file in glob.glob(main_dir.rstrip('/') + '/*/driving_log.csv'):
        with open(log_file) as csv_file:
            reader = csv.reader(csv_file)
            for line in reader:
                if len(line) > 0 and line[0] != "center":
                    for i in range(3):
                        line[i] = '/'.join((*log_file.split('/')[:-1],*line[i].lstrip(' ').split('/')[-2:]))
                    samples.append(line)
    return samples


def sample_generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                center_image = mpimg.imread(batch_sample[0]).astype(np.float32)
                left_image = mpimg.imread(batch_sample[1]).astype(np.float32)
                right_image = mpimg.imread(batch_sample[2]).astype(np.float32)

                steering_center = float(batch_sample[3])

                # create adjusted steering measurements for the side camera images
                correction = 0.2  # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                images.extend((left_image, right_image, center_image))
                angles.extend((steering_left, steering_right, steering_center))

                # flipped image
                images.append(np.fliplr(center_image))
                angles.append(-steering_center)

            yield shuffle(np.array(images), np.array(angles))
