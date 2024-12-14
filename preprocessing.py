import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

""" Section: Load Data """
(ds_train, ds_test), ds_info = tfds.load(
    'eurosat/rgb',                              # Three optical bands only; '.../all' contains 13 bands and is much larger
    split=['train[:80%]', 'train[80%:]'],       # Split the sole split called "train" into 80/20 train/test
    shuffle_files=True,
    with_info=True,
    as_supervised=True
)

# Get image and label shapes from info
image_shape = ds_info.features['image'].shape

print(image_shape)

""" Section: Normalize Data """
def normalize(image):
    image = tf.cast(image, tf.float32)  # Cast image to float32 first
    image = image / 255.0
    return image

def keep_label(label):
  return label

train_ds = (
    ds_train
    .map(lambda image, label: (normalize(image), label),
          num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .shuffle(buffer_size=1000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    ds_test
    .map(lambda image, label: (normalize(image), label),
          num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)