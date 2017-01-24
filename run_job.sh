#!/bin/bash
cp /job_files/theanorc /root/.theanorc
/job_files/train_action.py -e /job_files/dataset1_eval_3l_video.json
