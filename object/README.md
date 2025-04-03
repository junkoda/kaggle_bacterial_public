object
=======

Simple 2D object detection model

1: Preprocess training data

```bash
$ python3 train/prepare.py baseline.yml
```

2: Train model

Train

```bash
$ python3 train.py baseline.yml
```

Output will be created in `experiments/baseline`.

Use `--overwrite` to overwrite existing model.

```bash
$ python3 train/train.py baseline.yml --overwrite
```

