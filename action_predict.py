import numpy as np

import action_data_loader


def predict_patched(network, batch_index, batch_names, patches=10):
  """ Since one video contains the testing frames from exactly one batch, this
  averages the softmax distributions across an entire batch with patches, and
  produces a single prediction for the batch.
  Args:
    network: The network to use for the prediction.
    batch_index: The index of the batch to use.
    batch_names: A list of all the names for this batch.
    patches: The number of patches to average across.
  Returns:
    The predicted label from the network. """
  # Take care of averaging the patches using existing code.
  softmaxes = network.get_mean_softmax(batch_index, patches=patches)

  # Extract video names from the image names.
  video_names = []
  for name in batch_names:
    video_names.append(action_data_loader.get_video_name(name))

  # Now, we need to average the images for each video. Add all the labels first.
  averages = {}
  for name in video_names:
    if name not in averages:
      averages[name] = []

  # Now collect the softmaxes for each label.
  for label, softmax in zip(video_names, softmaxes):
    averages[label].append(softmax)

  # Finally, calculate the average for each of them.
  for name, data in averages.iteritems():
    np_data = np.asarray(data)
    mean = np.mean(np_data, axis=0)

    # The highest softmax score is our prediction for the video.
    sort = np.argsort(mean, axis=0)
    averages[name] = sort[-1]

  return averages
