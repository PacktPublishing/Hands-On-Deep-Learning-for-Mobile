import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import sys
sys.path.append('../')  # so that we can import libs from other chapters and utils
from utils import emnist_utils
######## GPU CONFIGS FOR RTX 2070 ###########
## Please ignore if not training on GPU ##
from tensorflow.keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True  # dont allocate all memory at once
config.gpu_options.allocator_type = 'BFC'  #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc
config.gpu_options.per_process_gpu_memory_fraction =
######## GPU CONFIGS FOR RTX 2070 ###########
## Please ignore if not training on GPU ##
from tensorflow.keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True  # dont allocate all memory at once
config.gpu_options.allocator_type = 'BFC'  #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc
config.gpu_options.per_process_gpu_memory_fraction = 1.0
session = InteractiveSession(config=config)
K.set_session(session)
# Load train and test data from Chapter 1 location
train = {}
test = {}

train['features'], train['labels'] = emnist_utils.read_emnist('../Chapter1/data/emnist-bymerge-train-images-idx3-ubyte.gz', '../Chapter1/data/emnist-bymerge-train-labels-idx1-ubyte.gz')
test['features'], test['labels'] = emnist_utils.read_emnist('../Chapter1/data/emnist-bymerge-test-images-idx3-ubyte.gz', '../Chapter1/data/emnist-bymerge-test-labels-idx1-ubyte.gz')
mappings = emnist_utils.map_emnist_labels("../Chapter1/data/emnist-bymerge-mapping.txt")
# Normalize the data
norm_train_features = keras.utils.normalize(train['features'])
norm_test_features = keras.utils.normalize(test['features'])

# one hot encode training labels
one_hot_train_labels = tf.one_hot(train['labels'], 47)
one_hot_test_labels = tf.one_hot(test['labels'], 47)
# (1) First the input layer
inputs = keras.Input(shape=(28,28,), name='emnist_inp')
x = layers.Reshape((28, 28, 1))(inputs)  # since images are gray scale, they have only one channel

# (2.1) Learn 128 different filters, each 3x3 in size, with valid pooling, and (1,1) stride size
x = layers.Conv2D(128, (3, 3), activation='relu', padding="same")(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding="same")(x)  # 2 convolutional layers
# (3.1) Pooling layer
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
# (4.1) Dropout Layer, dropping 20% of the connection
x = layers.Dropout(0.2)(x) 

# (2.2) Learn 64 different filters, each 3x3 in size, with valid pooling, and (1,1) stride size
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)  # 2 convolutional layers
# (3.2) Pooling layer
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
# (4.2) Dropout Layer, dropping 20% of the connection
x = layers.Dropout(0.2)(x) 

# (4) Dimensions after pooling are 5x5x64. The 28x28 image is now 5x5 with 64 filters
x = layers.Flatten()(x)

# (5) Traditional dense/FC layers to use these inputs for classification
# This part is similar to previous model
x = layers.Dense(256, activation='relu', name='dense_1')(x)
x = layers.Dense(128, activation='relu', name='dense_2')(x)

outputs = layers.Dense(47, activation='softmax', name='predictions')(x)

cnn2_dropout = keras.Model(inputs=inputs, outputs=outputs, name='cnn_model_dropout')
cnn2_dropout.summary()
import os, datetime
# Setup TensorBoard logging
# Make a subdirectory with date and time formatted string to keep track of runs
logdir = os.path.join(logs_base_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Register a callback with Keras to call after every epoch is completed
tensorboard_callback = keras.callbacks.TensorBoard(logdir, histogram_freq=1, write_images=True)

# Creates a file writer for the log directory. Used to log the first image
file_writer = tf.summary.create_file_writer(logdir)
import os, datetime

# sets the log directory as a subdirectory of the folder this python notebook is store in
logs_base_dir = "./logs"  

# If the directory does not exist, create it
os.makedirs(logs_base_dir, exist_ok=True)  

# Make a subdirectory with date and time formatted string to keep track of runs
logdir = os.path.join(logs_base_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Register a callback with Keras to call after every epoch is completed
tensorboard_callback = keras.callbacks.TensorBoard(logdir, histogram_freq=1, write_images=True)

# Creates a file writer for the log directory. Used to log the first image
file_writer = tf.summary.create_file_writer(logdir)
# Lets compile the model and train it
cnn2_dropout.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='categorical_crossentropy',
                                    metrics=['accuracy'])

# take a small part of the test set as a validation set
val_test_features = norm_test_features[:5000]  # there are 116K test samples
val_test_labels = one_hot_test_labels[:5000]

# Note the addition of the validation data and callback in the training loop
history = cnn2_dropout.fit(norm_train_features, one_hot_train_labels,
                           epochs=10, batch_size=32,
                                              validation_data=(val_test_features, val_test_labels),
                                                                 callbacks=[tensorboard_callback])
history = cnn2_dropout.fit(norm_train_features, one_hot_train_labels,
                           epochs=10, batch_size=128,
                                              validation_data=(val_test_features, val_test_labels)
                                                                        )

