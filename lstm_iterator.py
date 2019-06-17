from researchutils.chainer.iterators.decorable_serial_iterator import DecorableSerialIterator
from researchutils.chainer.iterators.decorable_multithread_iterator import DecorableMultithreadIterator

import chainer
import numpy


class LstmIterator(DecorableMultithreadIterator):
    def __init__(self, dataset, batch_size, k_step, init_steps, unroll_steps, mean_image, repeat=True, shuffle=True):
        self.k_step = k_step
        self.unroll_steps = unroll_steps
        self.init_steps = init_steps
        self.mean_image = mean_image
        end_index = len(dataset) - self.unroll_steps - 1
        super(LstmIterator, self).__init__(
            dataset, batch_size, repeat=repeat, shuffle=shuffle, decor_fun=self.prepare_training_data, end_index=end_index)

    def prepare_training_data(self, dataset, index):
        frame_start = index
        if self.k_step == 1:
            frame_stop = frame_start + self.unroll_steps
        else:
            frame_stop = frame_start + self.init_steps
        train_frames = [self.normalize_frame(train_data[0]) - self.mean_image for train_data in dataset[frame_start: frame_stop]]
        # print('train_frames shape: {}'.format(train_frames[0].shape))

        action_start = index + 1
        if self.k_step == 1:
            action_stop = action_start + self.unroll_steps
        else:
            action_stop = action_start + self.init_steps + self.k_step - 1
        train_actions = [train_data[1].astype(numpy.float32) for train_data in dataset[action_start: action_stop]]
        # print('train_actions shape: {}'.format(train_actions[0].shape))

        target_start = index + self.init_steps
        target_stop = action_stop
        target_frames = [self.normalize_frame(train_data[0])  - self.mean_image for train_data in dataset[target_start: target_stop]]
        # print('target frames shape: {}'.format(target_frames[0].shape))

        return ((train_frames, train_actions), target_frames)

    def normalize_frame(self, frame):
        return numpy.divide(frame, 255.0, dtype=numpy.float32)