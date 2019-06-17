from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json


def plot_results(train_result, eval_result, args):
    x_train, y_train = train_result[:, 0], train_result[:, 1]
    print('x_train: {}'.format(x_train))
    x_eval, y_eval = eval_result[:, 0], eval_result[:, 1]

    plt.plot(x_train, y_train, label='train', linewidth=1)
    plt.plot(x_eval, y_eval, label='evaluation', linewidth=1)
    plt.legend()

    x_lim = max(np.max(x_train), np.max(x_eval)) if args.x_lim == None else args.x_lim
    plt.xlim(0, x_lim)
    y_lim = max(np.max(y_train), np.max(y_eval)) if args.y_lim == None else args.y_lim
    plt.ylim(0, y_lim)

    plt.xlabel('iterations')
    plt.ylabel('loss')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style="sci",  axis="x", scilimits=(0,0))
  
    interval = 100000
    plt.xticks(np.arange(0, x_lim + interval, interval))

    plt.show()


def load_result(file_path, delimiter=' '):
    return np.loadtxt(file_path, delimiter=delimiter, skiprows=1)


def load_json(file_path):
    with open(file_path) as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-result-file', type=str, default='')
    parser.add_argument('--eval-result-file', type=str, default='')
    parser.add_argument('--x-lim', type=int, default=None)
    parser.add_argument('--y-lim', type=int, default=None)

    args = parser.parse_args()

    train_result = load_json(args.train_result_file)
    train_result = np.array([[data['iteration'], data['main/loss']] for data in train_result])
    eval_result = load_result(args.eval_result_file, delimiter='\t')

#    print('train: {}'.format(train_result))
#    print('eval: {}'.format(eval_result))
    plot_results(train_result, eval_result, args)


if __name__ == '__main__':
    main()
