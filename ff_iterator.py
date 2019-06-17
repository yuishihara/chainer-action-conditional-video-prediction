from researchutils.chainer.iterators.decorable_serial_iterator import DecorableSerialIterator
from researchutils.chainer.iterators.decorable_multithread_iterator import DecorableMultithreadIterator
import chainer
import numpy


class FFIterator(DecorableMultithreadIterator):
    def __init__(self, dataset, batch_size, skip_frames, k_step, mean_image, repeat=True, shuffle=True):
        self.skip_frames = skip_frames
        self.k_step = k_step
        self.mean_image = mean_image
        end_index = len(dataset) - self.k_step - self.skip_frames
        super(FFIterator, self).__init__(
            dataset, batch_size, repeat=repeat, shuffle=shuffle, n_threads=8, decor_fun=self.prepare_training_data, end_index=end_index)

    def prepare_training_data(self, dataset, index):
        # NOTE: mean image is already normalized
        frame_start = index
        frame_stop = index + self.skip_frames
        train_frames = [self.normalize_frame(
            train_data[0]) - self.mean_image for train_data in dataset[frame_start: frame_stop]]
        train_frames = numpy.array(train_frames)
        train_frames = numpy.reshape(train_frames, (self.skip_frames * self.mean_image.shape[0], self.mean_image.shape[1], self.mean_image.shape[2]))
        # print('train_frames shape: {}'.format(train_frames.shape))

        action_start = index + self.skip_frames
        action_stop = action_start + self.k_step
        train_actions = [train_data[1].astype(
            numpy.float32) for train_data in dataset[action_start: action_stop]]
        # print('train_actions shape: {}'.format(train_actions[0].shape))

        target_start = index + self.skip_frames
        target_stop = target_start + self.k_step
        target_frames = [self.normalize_frame(
            train_data[0]) - self.mean_image for train_data in dataset[target_start: target_stop]]
        # print('target frames shape: {}'.format(target_frames[0].shape))

        return ((train_frames, train_actions), target_frames)

    def normalize_frame(self, frame):
        return numpy.divide(frame, 255.0, dtype=numpy.float32)