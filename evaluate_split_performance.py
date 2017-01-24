#!/usr/bin/python


import json
import sys


""" Reads the JSON file that is split out during evaluation, and computes some
basic information about it. """


def compute_overall_score(eval_data):
  """ Computes an overall score for the split.
  Args:
    eval_data: The raw evaluation data for the split.
  Returns:
    The fraction of the videos predicted correctly. """
  # Figure out which predictions were correct and which weren't.
  all_videos = set()
  total_predictions = 0.0
  for outputs, _ in eval_data:
    for video, prediction in outputs.iteritems():
      if video in all_videos:
        # This is a duplicate prediction.
        continue
      all_videos.add(video)

      # Figure out what the actual label for this video is.
      label = int(video.split("_")[0][1:])
      prediction = int(prediction)
      if prediction == label:
        # Prediction is correct.
        total_predictions += 1.0

  # Calculate the average.
  return total_predictions / len(all_videos)

def main():
  if len(sys.argv) != 2:
    print "Usage: %s evaluation_file.json" % (sys.argv[0])
    sys.exit(1)

  print "Loading evaluation data..."
  evaluation_file = file(sys.argv[1])
  eval_data = json.load(evaluation_file)
  evaluation_file.close()

  overall_score = compute_overall_score(eval_data)
  print "Overall score: %f" % (overall_score)

if __name__ == "__main__":
  main()
