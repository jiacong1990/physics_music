"""Evaluate features for all neurons in the model use the sample."""
import argparse
from pathlib import Path
import subprocess

from keras.models import load_model


def evaluate(neuron_id, sample_file):
    evaluate_py = str(Path(__file__).parent / '../evaluate_feature.py')
    sample_dir = sample_file.parent / sample_file.stem
    subprocess.run(['python', evaluate_py,
                    '-m', str(sample_file.parent),
                    '-s', str(sample_file),
                    '--neuron_id', str(neuron_id),
                    '-o', str(sample_dir / '{}.pdf'.format(neuron_id))])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_file', type=Path, help='sample path')

    args = parser.parse_args()
    model_file = args.sample_file.parent / 'model.hdf5'
    model = load_model(str(model_file))
    nb_neuron = model.layers[0].output_shape[1]

    for i in range(nb_neuron):
        print('Evaluating neuron', i)
        evaluate(i, args.sample_file)
