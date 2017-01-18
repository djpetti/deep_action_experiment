#!/bin/bash
cp /job_files/theanorc /root/.theanorc
/job_files/train_action.py -e /job_files/dataset2_eval_3l.json
