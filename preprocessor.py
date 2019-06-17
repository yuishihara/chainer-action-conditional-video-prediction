from researchutils import files
from researchutils import arrays
from researchutils.image import preprocess
from researchutils.image import converter
from researchutils.image import viewer

import numpy as np
import os


def load_state_file(path):
    print('loading: {}'.format(path))
    states = files.load_pickle(path)
    frames = [state['frame'] for state in states]
    actions = [state['action'] for state in states]

    return frames, actions


def load_state_files(state_files):
    frames = []
    actions = []
    for state_file in state_files:
        loaded_frames, loaded_actions = load_state_file(state_file)
        frames.extend(loaded_frames)
        actions.extend(loaded_actions)

    return frames, actions


def save_mean_image(path, mean_image):
    filename = 'mean_image.pickle'
    files.create_dir_if_not_exist(path)
    files.save_pickle(os.path.join(path, filename), mean_image)


def create_mean_image_file(file_paths, save_path):
    frames, _ = load_state_files(file_paths)
    mean_image = preprocess.compute_mean_image(frames) / 255.0
    viewer.show_image(mean_image, title='mean_image')
    mean_image = converter.hwc2chw(mean_image)
    save_mean_image(save_path, mean_image)
    return mean_image


def preprocess_frames(frames):
    preprocessed_frames = [converter.hwc2chw(frame) for frame in frames]

    # for i in range(3):
    #    viewer.show_image(preprocessed_frames[0][i] + mean_image[0, :, :],
    #                      'preprocessed image. color channel -> {} (0: R, 1: G, 2: B) '.format(i))

    assert preprocessed_frames[0].shape == (3, 210, 160)
    return preprocessed_frames


def preprocess_actions(actions, num_actions):
    vectorized_actions = []
    for action in actions:
        vectorized_action = arrays.one_hot(
            indices=[action], shape=(num_actions, 1))
        vectorized_actions.append(vectorized_action)
    assert vectorized_actions[0].shape == (num_actions, 1)
    return vectorized_actions


def save_dataset(dataset, save_path):
    print('saving dataset')
    files.save_pickle(save_path, dataset)


def create_prefixed_dataset(file_paths, save_path, prefix):
    for index, file_path in enumerate(file_paths):
        dataset = create_dataset(file_path)
        save_dataset(dataset,
                     os.path.join(save_path, '{}{}.pickle'.format(prefix, index)))


def create_dataset(file_path):
    frames, actions = load_state_file(file_path)
    preprocessed_frames = preprocess_frames(frames)
    preprocessed_actions = preprocess_actions(actions, num_actions=3)
    assert len(preprocessed_frames) == len(preprocessed_actions)
    return dict(frames=preprocessed_frames, actions=preprocessed_actions)


def main():
    train_file_paths = [
        'states/{}.pickle'.format(index) for index in range(10000, 510000, 10000)]
    save_path = 'dataset'
    create_mean_image_file(train_file_paths, save_path)

    test_file_paths = [
        'states/{}.pickle'.format(index) for index in range(510000, 560000, 10000)]

    create_prefixed_dataset(train_file_paths, save_path, 'train')
    create_prefixed_dataset(test_file_paths, save_path, 'test')


if __name__ == '__main__':
    main()
