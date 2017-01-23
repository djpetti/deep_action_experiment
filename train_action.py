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
from rpinets.theano import data_loader

from six.moves import cPickle as pickle
import argparse
import json
import math
import os
import sys

from rpinets.common.data_manager import utils
from rpinets.theano.alexnet import AlexNet
from rpinets.theano import learning_rates

from action_data_loader import ActionTestingLoader
import action_alexnet


logger = logging.getLogger(__name__)


batch_size = 128
evaluation_batch_size = 150
# How many batches to have loaded into VRAM at once.
load_batches = 5
# How many batches to have loaded into VRAM at once when evaluating.
evaluation_load_batches = 4
# Shape of the input images.
image_shape = (240, 320, 3)
# Shape of the input patches.
patch_shape = (224, 224)

# Learning rate hyperparameters.
learning_rate = 0.01
momentum = 0.9
weight_decay = 0.00005

# Learning rate exponential decay hyperparameters.
decay_rate = 0.85
decay_steps = 10000

# Where to save the network.
save_file = "/job_files/action_frame_alexnet_split_2.pkl"
# Location of the original file that we should start training from.
alexnet_file = "/training_data/rpinets/alexnet.pkl"
synsets_save_file = "/job_files/synsets.pkl"
# Location of the dataset files.
dataset_files = "/training_data/action/split2_data/dataset"
# Location of the cache files.
cache_dir = "/training_data/action/split2_data/cache"


def train():
  """ Runs the action classifier training procedure. """
  data = data_loader.DataManagerLoader(batch_size, load_batches, image_shape,
                                       cache_dir, dataset_files,
                                       patch_shape=patch_shape)
  if not os.path.exists(synsets_save_file):
    logger.critical("Synset file '%s' not found!" % (synsets_save_file))
    sys.exit(1)
  logger.info("Loading synsets file...")
  data.load(synsets_save_file)

  train = data.get_train_set()
  test = data.get_test_set()
  cpu_labels = data.get_non_shared_test_set()[:]

  if os.path.exists(save_file):
    # Load the existing network and continue training.
    logger.info("Loading partially trained network '%s'..." % (save_file))
    network = AlexNet.load(save_file, train, test, batch_size,
                           learning_rate=learning_rate)
  else:
    # Load the pretrained AlexNet and work off of that.
    logger.info("Loading pretrained AlexNet '%s'..." % (alexnet_file))
    if not os.path.exists(alexnet_file):
      logger.critical("AlexNet file '%s' not found!" % (alexnet_file))
      sys.exit(1)

    # Setup exponential decay.
    use_lr = learning_rates.ExponentialDecay(decay_rate, decay_steps,
                                             learning_rate)

    network = AlexNet.load(alexnet_file, train, test, batch_size,
                           learning_rate=use_lr, train_layers=[5, 6, 7])

    # For exponential decay to work right, we're going to have to reset the
    # global step to zero.
    network.reset_global_step()
    # We're going to have to change the number of outputs.
    logger.info("Changing outputs from 1000 to 101...")
    network.replace_bottom_layers([], 101)

  logger.info("Starting training...")

  iterations = 0
  train_batch_index = 0
  test_batch_index = 0

  while iterations < 10000:
    logger.debug("Train index, size: %d, %d" % (train_batch_index,
                                                data.get_train_batch_size()))
    logger.debug("Test index, size: %d, %d" % (test_batch_index,
                                               data.get_test_batch_size()))

    # Swap in new data if we need to.
    if (train_batch_index + 1) * batch_size > data.get_train_batch_size():
      train_batch_index = 0
      logger.info("Getting train set.")
      train = data.get_train_set()
      logger.info("Got train set.")
    # Swap in new data if we need to.
    test_set_one_patch = data.get_test_batch_size() / 10
    if (test_batch_index + 1) * batch_size > test_set_one_patch:
      test_batch_index = 0
      logger.info("Getting test set.")
      test = data.get_test_set()
      cpu_labels = data.get_non_shared_test_set()[:]
      logger.info("Got test set.")

    if iterations % 100 == 0:
      # cpu_labels contains labels for every batch currently loaded in VRAM,
      # without duplicates for additional patches.
      label_index = test_batch_index * batch_size
      top_one, top_five = network.test(test_batch_index,
          cpu_labels[label_index:label_index + batch_size])
      logger.info("Step %d, testing top 1: %f, testing top 5: %f" % \
                  (iterations, top_one, top_five))

      test_batch_index += 1

    cost, rate, step = network.train(train_batch_index)
    logger.info("Training cost: %f, learning rate: %f, step: %d" % \
                (cost, rate, step))

    if iterations % 100 == 0:
      logger.info("Saving network...")
      network.save(save_file)
      # Save synset data as well.
      data.save(synsets_save_file)

    iterations += 1
    train_batch_index += 1

  data.exit_gracefully()

def evaluate_final(save_to):
  """ Evaluates the trained network.
  Args:
    save_to: Where to save the evaluation results. """
  data = ActionTestingLoader(evaluation_load_batches, image_shape, cache_dir,
                             dataset_files, patch_shape=patch_shape)
  if not os.path.exists(synsets_save_file):
    logger.critical("Synset file '%s' not found!" % (synsets_save_file))
    sys.exit(1)
  logger.info("Loading synsets file...")
  data.load(synsets_save_file)

  test = data.get_test_set()
  cpu_labels = data.get_non_shared_test_set()
  test_names = data.get_test_names()

  if not os.path.exists(save_file):
    logger.critical("Could not find saved network '%s'!" % (save_file))
    sys.exit(1)

  # Load the existing network and run evaluation.
  logger.info("Loading partially trained network '%s'..." % (save_file))
  network = AlexNet.load(save_file, None, test, evaluation_batch_size,
                         learning_rate=learning_rate)

  logger.info("Starting evaluation...")

  # Find out how many iterations we have to run.
  total_videos = data.get_num_videos()
  logger.debug("Have %d total videos in test set." % (total_videos))
  # Each video is represented by 25 frames in the batch.
  run_batches = float(total_videos) / (150 / 25) / evaluation_load_batches
  # We want to make sure every image is covered, which could result in partial
  # batches.
  run_batches = int(math.ceil(run_batches))
  logger.info("Running %d batches." % (run_batches))

  evaluation_data = []
  for i in range(0, run_batches):
    # Run on all the included batches.
    for batch_index in range(0, evaluation_load_batches):
      # Get the image names.
      truth_index = batch_index * batch_size
      names = test_names[truth_index:truth_index + batch_size]

      # We're going to use the prediction functionality to get the
      # raw prediction for this video.
      outputs = action_alexnet.predict_patched(network, batch_index, names)

      # Convert from neuron indices to actual labels.
      for video_name, output in outputs.iteritems():
        output = data.convert_ints_to_labels([output])[0]
        outputs[video_name] = output

      # Save the data.
      evaluation_data.append([outputs, names])
      logger.debug("Output: %s" % (outputs))

    # Swap in new data.
    logger.info("Getting test set.")
    test = data.get_test_set()
    cpu_labels = data.get_non_shared_test_set()
    test_names = data.get_test_names()
    logger.info("Got test set.")

  # Save the data to a JSON file.
  logger.info("Saving JSON file...")
  out_file = open(save_to, "w")
  json.dump(evaluation_data, out_file)
  out_file.close()
  logger.info("Done.")

  data.exit_gracefully()


def main():
  parser = argparse.ArgumentParser( \
      description="Train the action recognition network.")
  parser.add_argument("-e", "--evaluate", default=None,
                      help="Evaluate network and write output to this \
                            location.")
  args = parser.parse_args()

  if args.evaluate:
    # Evaluate the network
    logger.info("Running evaluation.")
    evaluate_final(args.evaluate)

  else:
    logger.info("Running training.")
    train()


if __name__ == "__main__":
  main()
