from ff_iterator import FFIterator
from ff_prediction_network import FeedForwardPredictionNetwork
from ff_prediction_evaluator import FFPredictionEvaluator

from lstm_iterator import LstmIterator
from lstm_prediction_network import LstmPredictionNetwork
from lstm_prediction_evaluator import LstmPredictionEvaluator
from lstm_updater import LstmUpdater

import dataset as ds

from researchutils.chainer.training.extensions.slack_report import SlackReport
from researchutils.chainer.functions import average_k_step_squared_error
from researchutils import files
from researchutils.image import converter, preprocess, viewer
import researchutils.chainer.serializers as serializers

import chainer
from chainer.backends import cuda
from chainer import training
from chainer import optimizers
import numpy as np
import os
import sys
import argparse

IMAGE_HEIGHT = 210
IMAGE_WIDTH = 160

chainer.config.autotune = True
chainer.backends.cuda.set_max_workspace_size(10000000000)


def print_shape(array, msg=''):
    print(msg + 'shape: {}'.format(array.shape))


def verify_batch(batch, mean_image, args):
    examples = ds.concat_examples_batch(batch)
    (train_frame, train_actions) = examples['input']
    target_frames = examples['target']
    print_shape(train_frame, 'train frame shape: ')
    if args.lstm:
        assert train_frame[0][0].shape == (
            args.color_channels,
            IMAGE_HEIGHT,
            IMAGE_WIDTH)
        assert train_actions.shape == (
            args.batch_size, args.unroll_steps, args.num_actions, 1)
        assert target_frames[0][0].shape == (
            args.color_channels,
            IMAGE_HEIGHT,
            IMAGE_WIDTH)
    else:
        assert train_frame.shape == (
            args.batch_size,
            args.color_channels * args.skip_frames,
            IMAGE_HEIGHT,
            IMAGE_WIDTH)
        assert train_actions.shape == (
            args.batch_size, args.k_step, args.num_actions, 1)
        assert target_frames[0][0].shape == (
            args.color_channels,
            IMAGE_HEIGHT,
            IMAGE_WIDTH)

    images = []
    titles = []
    if args.lstm:
        for i in range(args.init_steps):
            images.append(converter.chw2hwc(train_frame[0][i] + mean_image))
            titles.append('step={}, actions={}, action={}'.format(
                i, train_actions[0][0], np.argmax(train_actions[0][0])))
        for i in range(args.k_step):
            images.append(converter.chw2hwc(target_frames[0][i] + mean_image))
            titles.append('target frame kstep={}'.format(i + 1))
    else:
        for i in range(args.skip_frames):
            images.append(
                train_frame[0][i * args.color_channels] + mean_image[0])
            titles.append('channel={}, actions={}, action={}'.format(
                i, train_actions[0][0], np.argmax(train_actions[0][0])))
        for i in range(args.k_step):
            images.append(converter.chw2hwc(target_frames[0][i] + mean_image))
            titles.append('target frame kstep={}'.format(i + 1))

    viewer.show_images(images=images, titles=titles)


def train_model(model, train_iterator, test_iterator, args):
    outdir = files.prepare_output_dir(base_dir='result-{}-kstep{}'.format('lstm' if args.lstm else 'ff', args.k_step),
                                      args=args, time_format='')
    if args.lstm:
        model = LstmPredictionEvaluator(
            model, k_step=args.k_step, init_steps=args.init_steps, unroll_steps=args.unroll_steps)
    else:
        model = FFPredictionEvaluator(model, k_step=args.k_step)
    print('optimizer learning rate is set to: {}'.format(args.learning_rate))
    optimizer = optimizers.Adam(alpha=args.learning_rate)
    optimizer.setup(model)
    if args.lstm:
        updater = LstmUpdater(
            iterator=train_iterator,
            optimizer=optimizer,
            converter=ds.concat_examples_batch,
            device=args.gpu)
    else:
        updater = training.updaters.StandardUpdater(
            iterator=train_iterator,
            optimizer=optimizer,
            converter=ds.concat_examples_batch,
            device=args.gpu)
    evaluation_trigger = (args.test_interval, 'iteration')
    trainer = training.Trainer(updater,
                               stop_trigger=(args.max_iterations, 'iteration'),
                               out=outdir)
    # trainer.extend(training.extensions.Evaluator(iterator=test_iterator,
    #                                             target=model,
    #                                             converter=ds.concat_examples_batch,
    #                                             device=args.gpu),
    #               trigger=evaluation_trigger)
    trainer.extend(training.extensions.LogReport(log_name="training_results",
                                                 trigger=evaluation_trigger))
    trainer.extend(training.extensions.snapshot(
        filename='snapshot_iter-{.updater.iteration}'),
        trigger=evaluation_trigger)
    trainer.extend(training.extensions.snapshot_object(
        model.predictor,
        filename='model_iter-{.updater.iteration}'),
        trigger=evaluation_trigger)
    entries = ['iteration', 'main/loss',
               'validation/main/loss', 'elapsed_time']
    trainer.extend(training.extensions.PrintReport(entries=entries))
    trainer.extend(SlackReport(args.token, entries=entries, channel='learning-progress'),
                   trigger=evaluation_trigger)
    trainer.extend(training.extensions.dump_graph('main/loss'))

    if files.file_exists(args.snapshot_file):
        print('loading snapshot from: {}'.format(args.snapshot_file))
        serializers.load_snapshot(args.snapshot_file, trainer)

    trainer.run()


def prepare_iterator(train_data_files, test_data_files, mean_image, args):
    print('creating frame iterator for train data')
    train_dataset = ds.prepare_dataset(train_data_files)
    print('train dataset len: {}'.format(len(train_dataset)))
    assert train_dataset[0][0].shape == (
        args.color_channels, IMAGE_HEIGHT, IMAGE_WIDTH)
    assert train_dataset[0][1].shape == (args.num_actions, 1)

    if args.lstm:
        train_iterator = LstmIterator(
            dataset=train_dataset,
            batch_size=args.batch_size,
            k_step=args.k_step,
            init_steps=args.init_steps,
            unroll_steps=args.unroll_steps,
            mean_image=mean_image)
    else:
        train_iterator = FFIterator(
            dataset=train_dataset,
            batch_size=args.batch_size,
            skip_frames=args.skip_frames,
            k_step=args.k_step,
            mean_image=mean_image)

    print('creating frame iterator for test data')
    test_dataset = ds.prepare_dataset(test_data_files)
    print('test dataset len: {}'.format(len(test_dataset)))
    if args.lstm:
        test_iterator = LstmIterator(
            dataset=test_dataset,
            batch_size=args.batch_size,
            k_step=args.k_step,
            init_steps=args.init_steps,
            unroll_steps=args.unroll_steps,
            mean_image=mean_image,
            repeat=False,
            shuffle=False)
    else:
        test_iterator = FFIterator(
            dataset=test_dataset,
            batch_size=args.batch_size,
            skip_frames=args.skip_frames,
            k_step=args.k_step,
            mean_image=mean_image,
            repeat=False,
            shuffle=False)

    if args.verify_batch:
        batch = train_iterator.next()
        assert len(batch) == args.batch_size
        verify_batch(batch=batch,
                     mean_image=mean_image,
                     args=args)
        train_iterator.reset()

    return train_iterator, test_iterator


def join_path(directory, filename):
    return os.path.join(directory, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default='dataset')
    parser.add_argument('--mean-image-file', type=str,
                        default='mean_image.pickle')
    parser.add_argument('--token', type=str, default=None,
                        help='Slack client token')
    parser.add_argument('--max-iterations', type=int, default=1500000)
    parser.add_argument('--test-interval', type=int, default=10000)
    parser.add_argument('--color-channels', type=int, default=3)
    parser.add_argument('--skip-frames', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--k-step', type=int, default=1)
    parser.add_argument('--num-actions', type=int, default=3)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--unroll-steps', type=int, default=20)
    parser.add_argument('--init-steps', type=int, default=11)
    parser.add_argument('--train-file-num', type=int, default=10)
    parser.add_argument('--test-file-num', type=int, default=1)
    parser.add_argument('--verify-batch', action='store_true',
                        help='show created batch')
    parser.add_argument('--lstm', action='store_true',
                        help='train lstm network')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--model-file', type=str, default='')
    parser.add_argument('--snapshot-file', type=str, default='')

    args = parser.parse_args()

    print('loading mean image')
    mean_image = ds.load_mean_image(
        join_path(args.dataset_dir, args.mean_image_file))
    assert mean_image.shape == (args.color_channels, IMAGE_HEIGHT, IMAGE_WIDTH)

    if args.lstm:
        model = LstmPredictionNetwork()
    else:
        model = FeedForwardPredictionNetwork()

    if files.file_exists(args.model_file):
        print('loading model from: {}'.format(args.model_file))
        serializers.load_model(args.model_file, model)

    if not args.gpu < 0:
        model.to_gpu()

    train_data_files = [join_path(args.dataset_dir, 'train{}.pickle'.format(
        index)) for index in range(args.train_file_num)]
    test_data_files = [join_path(args.dataset_dir, 'test{}.pickle'.format(
        index)) for index in range(args.test_file_num)]

    train_iterator, test_iterator = prepare_iterator(
        train_data_files,
        test_data_files,
        mean_image,
        args)

    try:
        train_model(model, train_iterator, test_iterator, args=args)
    except:
        print('training finished with exception...')


if __name__ == '__main__':
    main()
