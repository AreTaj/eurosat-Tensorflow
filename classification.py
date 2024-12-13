import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

""" Section: Load Data """
(ds, ds_info) = tfds.load(
    'eurosat/rgb',      # Three optical bands only; '.../all' contains 13 bands and is much larger
    split=['train'],    # Only 'train' split present
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# Dataset itself is the first term of 'ds'
dataset = ds[0]

