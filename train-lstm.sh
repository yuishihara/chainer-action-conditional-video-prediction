#!/bin/bash

SLACK_TOKEN=`cat slack_token`

python main.py --lstm --init-steps=11 --unroll-steps=20 --train-file-num=50 --test-file-num=1 --num-actions=3 --k-step=1 --max-iterations=1000000 --batch-size=4 --learning-rate=1e-4 --token=$SLACK_TOKEN
python main.py --lstm --init-steps=11 --unroll-steps=13 --train-file-num=50 --test-file-num=1 --num-actions=3 --k-step=3 --max-iterations=1000000 --batch-size=4 --learning-rate=1e-5 --model-file=result-lstm-kstep1/model_iter-1000000 --token=$SLACK_TOKEN
python main.py --lstm --init-steps=11 --unroll-steps=15 --train-file-num=50 --test-file-num=1 --num-actions=3 --k-step=5 --max-iterations=1000000 --batch-size=4 --learning-rate=1e-5 --model-file=result-lstm-kstep3/model_iter-1000000 --token=$SLACK_TOKEN
