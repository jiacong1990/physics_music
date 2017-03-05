import argparse
import json
from pathlib import Path

from keras.models import load_model
from keras.models import Model
import matplotlib.pyplot as plt

from utils import to_one_hot_sequences


def main():
    # arguments are listed in alphabetic order
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-c', '--col', type=int, default=20,
                        help='number of columns')
    parser.add_argument('-m', '--model', type=str,
                        help='model directory')
    parser.add_argument('-n', '--neuron_id', type=int, default=0,
                        help='neuron index')
    parser.add_argument('-o', '--output_file', type=str, default=None,
                        help='output file name')
    parser.add_argument('-r', '--row', type=int, default=30,
                        help='number of rows')
    parser.add_argument('-s', '--sample', type=str,
                        help='sample file path')

    args = parser.parse_args()
    evaluate(args)


def evaluate(args):
    acts = neuron_activations(args)
    text = open(args.sample).read()
    plot_heat_map(acts, text[:args.row * args.col])


def neuron_activations(args):
    model_dir = Path(args.model)
    base_model = load_model(str(model_dir / 'model.hdf5'))
    model = Model(input=base_model.input, output=base_model.layers[0].output)
    _, seq_len, nb_char = model.input_shape

    text = open(args.sample).read()
    chars = json.load((model_dir / 'chars.json').open())
    x = [chars.index(c) + 1 for c in text]

    start = 1 - seq_len
    stop = start + args.row * args.col
    x = to_one_hot_sequences(x, nb_classes=nb_char, seq_len=seq_len,
                             start=start, stop=stop, step=1)
    y = model.predict(x)
    return y[:, args.neuron_id].reshape(args.row, args.col)


def plot_heat_map(data, labels):
    aspect = data.shape[0] / data.shape[1]
    plt.imshow(data, interpolation='nearest', aspect=1/aspect,
               cmap='coolwarm_r')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            k = i * data.shape[1] + j
            plt.annotate(repr(labels[k]).strip("'"),
                         xy=(j - .3, i + .4), xycoords='data')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
