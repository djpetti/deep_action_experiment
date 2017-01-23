import logging
import threading

from rpinets.common.data_manager import utils
from rpinets.theano import data_loader


logger = logging.getLogger(__name__)


def get_video_name(image_name):
  """ Extracts the video name from an image filename.
  Args:
    image_name: The name of the image file.
  Returns:
    The name of the video that the image belongs to. """
  video_name = image_name.split("_")[0:3]
  video_name = "%s_%s_%s" % (video_name[0], video_name[1], video_name[2])
  return video_name


class ActionTestingLoader(data_loader.DataManagerLoader):
  """ DataManagerLoader subclass customized for loading action recognition data.
  The main additional functionality is there to enable the special testing
  procedure that we have for this dataset. """

  def __init__(self, *args, **kwargs):
    """
    Args:
      Same as superclass, except the batch_size argument should not be
      specified. It will always be 25. """
    super(ActionTestingLoader, self).__init__(150, *args, **kwargs)

  def _init_image_getter(self):
    """ See superclass documentation. This version is also special in that it
    precomputes the set of testing frames. """
    # First, initialize the ImageGetter in the standard fashion.
    super(ActionTestingLoader, self)._init_image_getter()

    # Precompute the testing frames for all the videos in the dataset.
    images = self._image_getter.get_test_image_names()

    # The first task here is to figure out how many frames are in each video.
    video_lengths = {}
    for image in images:
      # Extract the components of the image ID.
      label, name = utils.split_img_id(image)

      video_name = (label, get_video_name(name))
      if video_name not in video_lengths:
        video_lengths[video_name] = 1
      else:
        video_lengths[video_name] += 1
    logger.debug("Got video lengths: %s" % (str(video_lengths)))

    self.__num_videos = len(video_lengths)

    # Now, we need to create the list of the frames that we will use for
    # testing.
    self.__testing_frames = []
    for video, length in video_lengths.iteritems():
      # Select 25 equally-spaced frames from each video.
      spacing = length / 25
      # Note that frames are all indexed from 1.
      start_offset = (length % 25) / 2 + 1

      frame_number = start_offset
      for _ in range(0, 25):
        label, video_name = video
        frame_name = "%s_f%03d.jpg" % (video_name, frame_number)
        # ImageGetter expects frames in (label, name) pairs.
        self.__testing_frames.append((label, frame_name))
        frame_number += spacing

    self.__testing_frames_index = 0

  def _init_loader_threads(self):
    """ Does not initialize a training loader thread, since we don't need it,
    and it's just a waste of time and memory. """
    logger.info("Not starting train loader. All train data requests will hang.")

    self._test_thread = threading.Thread(target=self._run_test_loader_thread)
    self._test_thread.start()

  def _load_raw_testing_batch(self):
    """ See superclass documentation. """
    # We want to fill the buffer up with only images that we're going to use.
    to_load = self._buffer_size
    load_frames = []
    if self.__testing_frames_index + self._buffer_size \
       > len(self.__testing_frames):
      # We wrapped here.
      logger.debug("Wrapped testing frames.")
      load_frames = self.__testing_frames[self.__testing_frames_index:]
      to_load -= len(load_frames)
      self.__testing_frames_index = 0

    index = self.__testing_frames_index
    load_frames.extend(self.__testing_frames[index:index + to_load])
    logger.debug("Loading %d images..." % (len(load_frames)))
    self.__testing_frames_index += to_load

    logger.debug("Testing frame index: %d" % (self.__testing_frames_index))

    # Load the actual data.
    return self._image_getter.get_specific_test_batch(load_frames)

  def get_num_videos(self):
    """ Returns:
      The total number of videos in use for testing. """
    return self.__num_videos
