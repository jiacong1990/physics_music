from keras.preprocessing.sequence import pad_sequences
import numpy as np


def to_one_hot_sequences(x, nb_classes=None, seq_len=30,
                         start=0, stop=None, step=0):
    if nb_classes is None:
        nb_classes = _guess_nb_classes(x)
    if stop is None:
        stop = len(x) - seq_len
    if step <= 0:
        step += seq_len
    seqs = [x[max(i, 0):i + seq_len] for i in range(start, stop, step)]
    seqs = pad_sequences(seqs, maxlen=seq_len, value=0)
    seqs = np.stack([to_one_hot_sequence(seq, nb_classes=nb_classes)
                     for seq in seqs])
    return seqs


def to_one_hot_sequence(x, nb_classes=None):
    if nb_classes is None:
        nb_classes = _guess_nb_classes(x)
    seq = np.zeros((len(x), nb_classes), dtype=np.bool)
    for i, j in enumerate(x):
        if j > 0:
            seq[i, j - 1] = 1
    return seq


def _guess_nb_classes(x):
    return len(set(x) - {0})
