import researchutils.chainer.serializers as serializers
from researchutils import files
from researchutils.image import viewer
from researchutils.image import converter
from ff_prediction_network import FeedForwardPredictionNetwork
from lstm_prediction_network import LstmPredictionNetwork
from chainer.dataset import to_device, concat_examples

import chainer
import chainer.functions as F
import argparse
import numpy as np

import dataset as ds


def normalize_frame(frame):
    return np.divide(frame, 255.0, dtype=np.float32)


def animate_dataset(dataset, args):
    initial_frame = args.initial_frame
    last_frame = args.last_frame
    frames = [converter.chw2hwc(frame) for frame in dataset['frames']]
    viewer.animate(frames[initial_frame:last_frame], titles=['dataset'])


def animate_lstm(model, frames, actions, mean_image, args):
    with chainer.no_backprop_mode():
        for step in range(args.init_steps - 1):
            print('next_index: {}'.format(step))
            frame = frames[step]
            normalized_frame = normalize_frame(frame)
            input_frame = normalized_frame - mean_image
            input_frame = to_device(
                device=args.gpu, x=input_frame.reshape((-1, ) + input_frame.shape))
            print('input_frame shape: {}'.format(input_frame.shape))
            input_action = actions[step + 1].astype(np.float32)
            input_action = to_device(
                device=args.gpu, x=input_action.reshape((-1, 3, 1)))

            model((input_frame, input_action))

    predicted_frames = []
    ground_truths = []

    frame = frames[args.init_steps - 1]
    normalized_frame = normalize_frame(frame)
    next_frame = normalized_frame - mean_image
    next_frame = to_device(
        device=args.gpu, x=input_frame.reshape((-1, ) + next_frame.shape))

    with chainer.no_backprop_mode():
        for next_index in range(args.init_steps - 1, len(frames) - 1):
            print('next_index: {}'.format(next_index))
            input_frame = next_frame
            input_action = actions[next_index].astype(np.float32)
            input_action = to_device(
                device=args.gpu, x=input_action.reshape((-1, 3, 1)))

            predicted_frame = model((input_frame, input_action))

            next_frame = chainer.Variable(predicted_frame.array)
            # Keep predicted image
            predicted_frame.to_cpu()
            predicted_frames.append(converter.chw2hwc(
                predicted_frame.data[0] + mean_image))
            ground_truth = frames[next_index]
            ground_truths.append(converter.chw2hwc(ground_truth))

    return ground_truths, predicted_frames


def animate_ff(model, frames, actions, mean_image, args):
    initial_frames = np.concatenate(
        [normalize_frame(frame) - mean_image for frame in frames[0:args.skip_frames]])
    initial_action = actions[args.skip_frames - 1].astype(np.float32)

    input_frames, input_action = concat_examples(
        batch=[(initial_frames, initial_action)], device=args.gpu)
    print('input_frames shape: {}'.format(input_frames.shape))

    predicted_frames = []
    ground_truths = []
    with chainer.no_backprop_mode():
        for next_index in range(args.skip_frames, len(frames)):
            print('next_index: {}'.format(next_index))
            predicted_frame = model((input_frames, input_action))
            next_frames = F.concat((input_frames, predicted_frame), axis=1)[
                :, 3:, :, :]
            next_action = actions[next_index].astype(np.float32)

            input_frames = next_frames
            input_action = to_device(
                device=args.gpu, x=next_action.reshape((-1, 3, 1)))
            # print('next_frames shape: {}'.format(input_frames.shape))
            # print('next_action shape: {}'.format(input_action.shape))
            # Keep predicted image
            predicted_frame.to_cpu()
            predicted_frames.append(converter.chw2hwc(
                predicted_frame.data[0] + mean_image))
            ground_truth = frames[next_index]
            ground_truths.append(converter.chw2hwc(ground_truth))

    return ground_truths, predicted_frames


def animate_predictions(model, dataset, mean_image, args):
    initial_frame = args.initial_frame
    last_frame = args.last_frame
    frames = dataset['frames'][initial_frame:last_frame]
    actions = dataset['actions'][initial_frame + 1:last_frame + 1]

    if args.lstm:
        ground_truths, predicted_frames = animate_lstm(
            model, frames, actions, mean_image, args)
    else:
        ground_truths, predicted_frames = animate_ff(
            model, frames, actions, mean_image, args)

    viewer.animate(ground_truths, predicted_frames,
                   titles=['ground truth', 'predicted'],
                   fps=10,
                   repeat=True,
                   save_mp4=True,
                   auto_close=True)


def load_dataset(dataset_paths):
    frames = None
    actions = None
    for path in dataset_paths:
        print('loading data from file: {}'.format(path))
        dataset = files.load_pickle(path)
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

    print('creating data set of {}'.format(dataset_paths))
    assert len(frames) == len(actions)
    return dict(frames=frames, actions=actions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default='')
    parser.add_argument('--mean-image-file', type=str,
                        default='mean_image.pickle')
    parser.add_argument('--dataset-file', type=str,
                        default='100000.pickle')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--skip-frames', type=int, default=4)
    parser.add_argument('--initial-frame', type=int, default=0)
    parser.add_argument('--last-frame', type=int, default=10000)
    parser.add_argument('--init-steps', type=int, default=11)
    parser.add_argument('--show-dataset', action='store_true')
    parser.add_argument('--show-prediction', action='store_true')
    parser.add_argument('--show-mean-image', action='store_true')
    parser.add_argument('--show-sample-frame', action='store_true')
    parser.add_argument('--lstm', action='store_true')

    args = parser.parse_args()

    if args.show_dataset:
        dataset = load_dataset([args.dataset_file])
        animate_dataset(dataset, args)

    if args.show_prediction:
        dataset = load_dataset([args.dataset_file])
        if args.lstm:
            model = LstmPredictionNetwork()
        else:
            model = FeedForwardPredictionNetwork()
        serializers.load_model(args.model_file, model)
        if not args.gpu < 0:
            model.to_gpu()

        mean_image = ds.load_mean_image(args.mean_image_file)
        animate_predictions(model, dataset, mean_image, args)

    if args.show_mean_image:
        mean_image = ds.load_mean_image(args.mean_image_file)
        viewer.show_image(converter.chw2hwc(mean_image), title='mean image')

    if args.show_sample_frame:
        dataset = load_dataset([args.dataset_file])
        frame = dataset['frames'][args.initial_frame]
        viewer.show_image(converter.chw2hwc(frame), title='frame')


if __name__ == '__main__':
    main()
