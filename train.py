import argparse
from collections import namedtuple
import json
import os

from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import SimpleRNN
from keras.models import load_model
from keras.models import Sequential
import numpy as np


def main():
    # arguments are listed in alphabetic order
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout fraction')
    parser.add_argument('--hidden_dim', type=int, default=75,
                        help='hidden layer dimension')
    parser.add_argument('--input_file', type=str, default='data/input.txt',
                        help='path to the input file')
    parser.add_argument('--nb_epoch', type=int, default=1,
                        help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='rmsprop',
                        help='name of the optimizer')
    parser.add_argument('--output_dir', type=str, default='model-test',
                        help='output directory')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume from saved model')
    parser.add_argument('--seq_length', type=int, default=30,
                        help='sequence length')
    parser.add_argument('--seq_stride', type=int, default=0,
                        help='sequence stride')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='fraction of the validation data')

    args = parser.parse_args()
    # make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    data = prepare_data(args)
    model_file = os.path.join(args.output_dir, 'model.hdf5')
    if args.resume and os.path.exists(model_file):
        print('Resume from saved model')
        model = load_model(model_file)
    else:
        model = build_model(data, args)
    model.summary()
    train(model, data, args)


def prepare_data(args):
    # read text from file
    text = open(args.input_file).read()

    # save characters
    chars = sorted(list(set(text)))

    with open(os.path.join(args.output_dir, 'chars.json'), 'w') as f:
        json.dump(chars, f)

    # prepare input character sequences and target characters
    sequences = []
    targets = []
    if args.seq_stride > 0:
        step = args.seq_stride
    else:
        step = args.seq_length - args.seq_stride
    for i in range(0, len(text) - args.seq_length, step):
        sequences.append(text[i: i + args.seq_length])
        targets.append(text[i + args.seq_length])

    # convert characters to one-hot vectors
    c2i = dict(zip(chars, range(len(chars))))
    x = np.zeros((len(sequences), args.seq_length, len(chars)), dtype=np.bool)
    y = np.zeros((len(sequences), len(chars)), dtype=np.bool)
    for i, sequence in enumerate(sequences):
        for t, char in enumerate(sequence):
            x[i, t, c2i[char]] = 1
        y[i, c2i[targets[i]]] = 1

    # print diagnostic information and return
    nb_seq, seq_len, nb_char = x.shape
    print('-' * 10)
    print('number of sequences:', nb_seq)
    print('sequence length:', seq_len)
    print('number of characters:', nb_char)
    print('-' * 10)
    Data = namedtuple('Data', ['x', 'y'])
    data = Data(x=x, y=y)
    return data


def build_model(data, args):
    # build model according to data dimensions
    nb_samples, timesteps, features = data.x.shape
    model = Sequential()
    model.add(SimpleRNN(args.hidden_dim, input_shape=(timesteps, features)))
    model.add(Dropout(args.dropout))
    model.add(Dense(features))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=args.optimizer)
    return model


def train(model, data, args):
    # set up callbacks to record model and history
    model_file = os.path.join(args.output_dir, 'model.hdf5')
    history_file = os.path.join(args.output_dir, 'history.csv')
    model_checkpoint = ModelCheckpoint(model_file)
    csv_logger = CSVLogger(history_file, append=args.resume)

    # train
    model.fit(data.x, data.y, nb_epoch=args.nb_epoch,
              validation_split=args.val_split,
              callbacks=[model_checkpoint, csv_logger])


if __name__ == '__main__':
    main()
