# physics_music

## Example

To train a new model

```bash
python train.py --hidden_dim 75 --dropout 0.2 --optimizer rmsprop --output_dir model-test --nb_epoch 2
```

To resume from a saved model

```bash
python train.py --resume --output_dir model-test --nb_epoch 2
```

To sample using a saved model

```bash
python sample.py -m model-test -t 1 -l 100 -o model-test/sample.txt
```

### Specialized Scripts

To generate 6 sample music pieces

```bash
python scripts/generate.py model-test
```

## Usage

```bash
python train.py --help
```

```
usage: train.py [-h] [--dropout DROPOUT] [--hidden_dim HIDDEN_DIM]
                [--input_file INPUT_FILE] [--nb_epoch NB_EPOCH]
                [--optimizer OPTIMIZER] [--output_dir OUTPUT_DIR] [--resume]
                [--seq_length SEQ_LENGTH] [--seq_stride SEQ_STRIDE]
                [--val_split VAL_SPLIT]

optional arguments:
  -h, --help            show this help message and exit
  --dropout DROPOUT     dropout fraction (default: 0.2)
  --hidden_dim HIDDEN_DIM
                        hidden layer dimension (default: 75)
  --input_file INPUT_FILE
                        path to the input file (default: data/input.txt)
  --nb_epoch NB_EPOCH   number of epochs (default: 1)
  --optimizer OPTIMIZER
                        name of the optimizer (default: rmsprop)
  --output_dir OUTPUT_DIR
                        output directory (default: model-test)
  --resume              resume from saved model (default: False)
  --seq_length SEQ_LENGTH
                        sequence length (default: 30)
  --seq_stride SEQ_STRIDE
                        sequence stride (default: 0)
  --val_split VAL_SPLIT
                        fraction of the validation data (default: 0.2)
```

```bash
python sample.py --help
```

```
usage: sample.py [-h] [-l LENGTH] [-m MODEL] [-o OUTPUT_FILE] [-p PRIME]
                 [-t TEMPERATURE] [-u UNTIL]

optional arguments:
  -h, --help            show this help message and exit
  -l LENGTH, --length LENGTH
                        maximum sampling length (default: 2000)
  -m MODEL, --model MODEL
                        model directory (default: None)
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        output file name (default: None)
  -p PRIME, --prime PRIME
                        prime sequence (default: <start> X:)
  -t TEMPERATURE, --temperature TEMPERATURE
                        higher temperature increases diversity (default: 1)
  -u UNTIL, --until UNTIL
                        stop sampling when the until sequence appears
                        (default: <end>)

```
