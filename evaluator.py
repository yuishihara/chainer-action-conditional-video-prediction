from ff_iterator import FFIterator
from ff_prediction_evaluator import FFPredictionEvaluator
from ff_prediction_network import FeedForwardPredictionNetwork

from lstm_iterator import LstmIterator
from lstm_prediction_evaluator import LstmPredictionEvaluator
from lstm_prediction_network import LstmPredictionNetwork

from dataset import concat_examples_batch

import researchutils.files as files
import researchutils.chainer.serializers as serializers
import dataset as ds

import chainer
import argparse

import os


def evaluate_model(model, iterator, args):
    if args.lstm:
        model = LstmPredictionEvaluator(
            model, k_step=args.k_step, init_steps=args.init_steps, unroll_steps=args.unroll_steps)
    else:
        model = FFPredictionEvaluator(model, k_step=args.k_step)
    iterator.reset()
    loss = 0
    batch_num = 0
    for batch in iterator:
        with chainer.no_backprop_mode():
            examples = concat_examples_batch(batch, device=args.gpu)
            batch_loss = model(**examples)
            loss += batch_loss
            batch_num += 1
        if iterator.is_new_epoch:
            break
    evaluation_loss = loss / batch_num
    print('evaluation loss: {}'.format(evaluation_loss))

    return evaluation_loss


def prepare_iterator(dataset, mean_image, args):
    if args.lstm:
        iterator = LstmIterator(
            dataset=dataset,
            batch_size=args.batch_size,
            k_step=args.k_step,
            init_steps=args.init_steps,
            unroll_steps=args.unroll_steps,
            mean_image=mean_image,
            repeat=False,
            shuffle=False)
    else:
        iterator = FFIterator(
            dataset=dataset,
            batch_size=args.batch_size,
            skip_frames=args.skip_frames,
            k_step=args.k_step,
            mean_image=mean_image,
            repeat=False,
            shuffle=False)
    return iterator


def join_path(directory, filename):
    return os.path.join(directory, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lstm', action='store_true',
                        help='evaluate lstm network')
    parser.add_argument('--unroll-steps', type=int, default=20)
    parser.add_argument('--init-steps', type=int, default=11)
    parser.add_argument('--mean-image-file', type=str, default='')
    parser.add_argument('--model-dir', type=str, default='')
    parser.add_argument('--dataset-dir', type=str, default='')
    parser.add_argument('--skip-frames', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--k-step', type=int, default=1)
    parser.add_argument('--iteration-num', type=int, default=1000000)
    parser.add_argument('--dataset-num', type=int, default=1)
    parser.add_argument('--output-file', type=str,
                        default='evaluation_results')
    args = parser.parse_args()

    chainer.backends.cuda.set_max_workspace_size(10000000000)
    chainer.config.autotune = True

    dataset_files = [join_path(args.dataset_dir, 'test{}.pickle'.format(
        index)) for index in range(args.dataset_num)]
    dataset = ds.prepare_dataset(dataset_files)
    mean_image = ds.load_mean_image(args.mean_image_file)

    iterator = prepare_iterator(dataset, mean_image, args)

    losses = []
    if args.lstm:
        model = LstmPredictionNetwork()
    else:
        model = FeedForwardPredictionNetwork()
    if not args.gpu < 0:
        model.to_gpu()

    with open(args.output_file, "w") as f:
        f.write('iteration\tloss\n')
        for iteration in range(830000, args.iteration_num + 1, 10000):
            model_file = join_path(
                args.model_dir, 'model_iter-{}'.format(iteration))
            print('loading model: {}'.format(model_file))
            if files.file_exists(model_file):
                serializers.load_model(model_file, model)
                loss = evaluate_model(model, iterator, args)
                losses.append(dict(iteration=iteration, loss=loss))
                f.write('{}\t{}\n'.format(iteration, loss.data))
            else:
                print('Model {} not found. skipping evaluation.'.format(model_file))

    print('losses: {}'.format(losses))


if __name__ == '__main__':
    main()
