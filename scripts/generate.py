"""Generate 6 sample music pieces."""
import argparse
from pathlib import Path
import subprocess


def generate(model_dir, temperature, i):
    sample_py = str((Path(__file__).parent / '../sample.py'))
    sample_name = 'sample-t{:.1f}-{}.txt'.format(temperature, i)
    subprocess.run(['python', sample_py, '-m', model_dir,
                    '-t', str(temperature),
                    '-o', str(Path(model_dir) / sample_name)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str, help='model directory')

    args = parser.parse_args()
    for temperature in [1., 2., 0.5]:
        for i in range(2):
            generate(args.model_dir, temperature, i)
