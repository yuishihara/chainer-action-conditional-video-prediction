from researchutils import files

from chainer.dataset import to_device, concat_examples

import numpy as np


def load_mean_image(path):
    return files.load_pickle(path).astype(np.float32)


def prepare_dataset(dataset_files):
    frames = None
    actions = None
    for data_file in dataset_files:
        print('loading data from file: {}'.format(data_file))
        dataset = files.load_pickle(data_file)
        loaded_frames = dataset['frames']
        loaded_actions = dataset['actions']
        if frames is None:
            frames = loaded_frames
        else:
            frames.extend(loaded_frames)

        if actions is None:
            actions = loaded_actions
        else:
            actions.extend(loaded_actions)

    print('creating data set of {}'.format(dataset_files))
    assert len(frames) == len(actions)
    dataset = list(zip(frames, actions))
    return dataset


def concat_examples_batch(batch, device=None):
    train_frame = []
    train_actions = []
    target_frames = []
    for ((frame, actions), target) in batch:
        train_frame.append(frame)
        train_actions.append(actions)
        target_frames.append(target)
    train_frame = np.array(train_frame)
    train_actions = np.array(train_actions)
    target_frames = np.array(target_frames)
    if not device is None:
        train_frame = to_device(device, train_frame)
        train_actions = to_device(device, train_actions)
        target_frames = to_device(device, target_frames)
    # print('train frame shape: {}, train actions shape: {}, target_frames shape: {}'.format(train_frame.shape, train_actions.shape, target_frames.shape))
    return dict(input=(train_frame, train_actions), target=target_frames)
