#!/bin/bash

python analyzer.py --model-file=result-ff-kstep1/model_iter-1000000 --mean-image-file=dataset/mean_image.pickle --dataset-file=dataset/test0.pickle --initial-frame=0 --last-frame=300 --show-prediction
mv anim.mp4 result-ff-kstep1/
python analyzer.py --model-file=result-ff-kstep3/model_iter-1000000 --mean-image-file=dataset/mean_image.pickle --dataset-file=dataset/test0.pickle --initial-frame=0 --last-frame=300 --show-prediction
mv anim.mp4 result-ff-kstep3/
python analyzer.py --model-file=result-ff-kstep5/model_iter-1000000 --mean-image-file=dataset/mean_image.pickle --dataset-file=dataset/test0.pickle --initial-frame=0 --last-frame=300 --show-prediction
mv anim.mp4 result-ff-kstep5/