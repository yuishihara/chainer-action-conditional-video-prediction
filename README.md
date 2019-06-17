# About

This is a reproduction code of "Action Conditional Video Prediction using Deep Networks in Atari Games" proposed by Oh et al.

[See here](https://arxiv.org/abs/1507.08750) for original paper

Entire code is written in python using [chainer](https://chainer.org/) as backend deep learning framework

# Results

## Feedforward version

It took about 5 days in total with Quadro P6000 to train the network.
The network was trained for 1M iterations for each k's(1, 3 and 5) mentioned in original paper.
Adam is used in this reproduction code instead of RMSProp according to the suggestion ([see here](https://github.com/junhyukoh/nips2015-action-conditional-video-prediction))
by the original author.

### Video

![Feedforward result](https://raw.githubusercontent.com/yuishihara/Reproductions/master/action-conditional-video-prediction/trained_results/feedforward/result.gif)

## LSTM version

It took about 1wk in total with Quadro P6000 to train the network.
Trained conditions are same as Feedforward version descrived above.
The video shown below are final result of 1step training after 1M steps.
3 and 5 steps training seems to be overfitted in my environment and the results were poorer than 1step training.

### Video

![LSTM result](https://raw.githubusercontent.com/yuishihara/Reproductions/master/action-conditional-video-prediction/trained_results/lstm/result.gif)