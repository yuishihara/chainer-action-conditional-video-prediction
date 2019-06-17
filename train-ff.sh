#!/bin/bash

SLACK_TOKEN=`cat slack_token`

python main.py --train-file-num=50 --test-file-num=1 --num-actions=3 --k-step=1 --max-iterations=1500000 --batch-size=32 --learning-rate=1e-4 --token=$SLACK_TOKEN
python main.py --train-file-num=50 --test-file-num=1 --num-actions=3 --k-step=3 --max-iterations=1000000 --batch-size=8 --learning-rate=1e-5 --model-file=result-ff-kstep1/model_iter-1000000 --token=$SLACK_TOKEN
python main.py --train-file-num=50 --test-file-num=1 --num-actions=3 --k-step=5 --max-iterations=1000000 --batch-size=8 --learning-rate=1e-5 --model-file=result-ff-kstep3/model_iter-1000000 --token=$SLACK_TOKEN