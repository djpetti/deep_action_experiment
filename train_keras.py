#!/usr/bin/python

""" Train the action recognizer. """

import logging

def _configure_logging():
  """ Configure logging handlers. """
  # Configure root logger.
  root = logging.getLogger()
  root.setLevel(logging.DEBUG)
  file_handler = logging.FileHandler("/job_files/run_tests.log")
  file_handler.setLevel(logging.DEBUG)
  stream_handler = logging.StreamHandler()
  stream_handler.setLevel(logging.WARNING)
  formatter = logging.Formatter("%(name)s@%(asctime)s: " +
      "[%(levelname)s] %(message)s")
  file_handler.setFormatter(formatter)
  stream_handler.setFormatter(formatter)
  root.addHandler(file_handler)
  root.addHandler(stream_handler)

# Some modules need a logger to be configured immediately.
_configure_logging()


# This forks a lot of processes, so we want to import it as soon as possible,
# when there is as little memory as possible in use.
from rpinets.myelin import data_loader

from six.moves import cPickle as pickle
import argparse
import json
import math
import os
import sys

from keras.models import Model
import keras.layers as layers
import keras.optimizers as optimizers

from rpinets.myelin import utils

from action_data_loader import ActionTestingLoader


logger = logging.getLogger(__name__)


batch_size = 256
eval_batch_size = 150
# How many batches to have loaded into VRAM at once.
load_batches = 4
# How many batches to have loaded into VRAM at once when evaluating.
eval_load_batches = 4
# Shape of the input images.
image_shape = (240, 320, 3)
# Shape of the input patches.
patch_shape = (224, 224)

# How many iterations to train for.
iterations = 320000

# Learning rate hyperparameters.
learning_rate = 0.01
momentum = 0.9
# Learning rate decay.
decay = learning_rate / iterations

# Learning rate exponential decay hyperparameters.
decay_rate = 0.85
decay_steps = 10000

# Where to save the network.
save_file = "/job_files/action_frame_alexnet_split_1.pkl"
synsets_save_file = "/job_files/synsets.pkl"
# Location of the original file that we should start training from.
alexnet_file = "/training_data/action/alexnet_m_2048.pkl"
alexnet_synsets_file = "/training_data/rpinets/synsets.pkl"
# Location of the dataset files.
dataset_files = "/training_data/action/split1_data/dataset"
alexnet_dataset_files = "/training_data/rpinets/ilsvrc16_dataset"
# Location of the cache files.
cache_dir = "/training_data/action/split1_data/cache"
alexnet_cache_dir = "/training_data/rpinets/cache"
# Location of synset data.
synset_location = "/training_data/rpinets/synsets"
synset_list = "/job_files/ilsvrc_synsets.txt"


def pretrain():
  """ Pretrains the network on the ILSVRC dataset. """

  data = data_loader.ImagenetLoader(batch_size, load_batches,
                                    alexnet_cache_dir, alexnet_dataset_files,
                                    synset_location, synset_list)
  if os.path.exists(alexnet_synsets_file):
    logger.info("Loading synsets file...")
    data.load(alexnet_synsets_file)

  # Make the network.
  logger.info("Building network.")

  # Shape of input data.
  input_shape = (patch_shape[0], patch_shape[1], 3)
  inputs = layers.Input(shape=input_shape)

  values = layers.Convolution2D(96, 7, 7, activation="relu", border_mode="same",
                                subsample=(2, 2))(inputs)
  values = layers.BatchNormalization()(values)
  values = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                               border_mode="same")(values)
  values = layers.Convolution2D(256, 5, 5, activation="relu",
                                border_mode="same", subsample=(2, 2))(values)
  values = layers.BatchNormalization()(values)
  values = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                               border_mode="same")(values)

  values = layers.Convolution2D(512, 3, 3, activation="relu",
                                border_mode="same", subsample=(2, 2))(values)
  values = layers.Convolution2D(512, 3, 3, activation="relu",
                                border_mode="same", subsample=(2, 2))(values)
  values = layers.Convolution2D(512, 3, 3, activation="relu",
                                border_mode="same", subsample=(2, 2))(values)

  values = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                               border_mode="same")(values)

  values = layers.Flatten()(values)

  values = layers.Dense(4096, activation="relu")(values)
  values = layers.Dropout(0.5)(values)
  values = layers.Dense(2048, activation="relu")(values)
  values = layers.Dropout(0.5)(values)

  predictions = layers.Dense(1000, activation="softmax")(values)

  model = Model(input=inputs, output=predictions)
  optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum,
                             decay=decay)
  model.compile(optimizer, "sparse_categorical_crossentropy", metrics=["accuracy"])

  # Run the training loop.
  logger.info("Starting training...")

  for i in range(0, iterations):
    # Get a new chunk of training data.
    training_data, training_labels = data.get_train_set()
    # Train the model.
    model.fit(training_data, training_labels, nb_epoch=1,
              batch_size=batch_size)

    if not i % 1000:
      # Evaluate the model on testing data.
      testing_data, testing_labels = data.get_test_set()

      # We want to average accross all ten patches.
      loss, accuracy = model.evaluate(testing_data, testing_labels,
                                      batch_size=batch_size)
      print accuracy
      logger.info("Testing accuracy: %d" % (accuracy))


def main():
  pretrain()


if __name__ == "__main__":
  main()
