#!/bin/bash

python evaluator.py --init-steps=11 --unroll-steps=20 --lstm --batch-size=4 --dataset-dir=dataset/ --model-dir=result-lstm-kstep1/ --k-step=1 --mean-image-file=dataset/mean_image.pickle --dataset-num=5
# mv evaluation_results result-lstm-kstep1/
# python evaluator.py --init-steps=11 --unroll-steps=13 --lstm --batch-size=4 --dataset-dir=dataset/ --model-dir=result-lstm-kstep3/ --k-step=3 --mean-image-file=dataset/mean_image.pickle --dataset-num=5
# mv evaluation_results result-lstm-kstep3/
# python evaluator.py --init-steps=11 --unroll-steps=15 --lstm --batch-size=4 --dataset-dir=dataset/ --model-dir=result-lstm-kstep5/ --k-step=5 --mean-image-file=dataset/mean_image.pickle --dataset-num=5
# mv evaluation_results result-lstm-kstep5/

# python evaluator.py --dataset-dir=dataset/ --model-dir=result-ff-kstep1/ --k-step=1 --mean-image-file=dataset/mean_image.pickle --dataset-num=5
# mv evaluation_results result-ff-kstep1/
# python evaluator.py --batch-size=8 --dataset-dir=dataset/ --model-dir=result-ff-kstep3/ --k-step=3 --mean-image-file=dataset/mean_image.pickle --dataset-num=5
# mv evaluation_results result-ff-kstep3/
# python evaluator.py --batch-size=8  --dataset-dir=dataset/ --model-dir=result-ff-kstep5/ --k-step=5 --mean-image-file=dataset/mean_image.pickle --dataset-num=5
# mv evaluation_results result-ff-kstep5/
