import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
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

with open('exploratory_output.txt', 'w') as f:  # Open file for writing
  # Get and show dataset information
  print(f"Dataset features: {ds_info.features}\n", file=f)

  # Get and show type of 'dataset' (actual dataset), not 'ds' (list containing dataset)
  print(f"Type: {type(dataset)}\n", file=f)

  # Explore features
  print("___Features of examples___", file=f)
  for example in dataset.take(5):
    image, label = example
    print(f"Image shape: {image.shape}", file=f)
    print(f"Label: {label}", file=f)

  # Explore class distribution
  class_names = ds_info.features['label'].names
  class_counts = {}
  for example in dataset:
    _, label = example
    class_counts[class_names[label.numpy()]] = class_counts.get(class_names[label.numpy()], 0) + 1
  print(f"\nClass distribution: {class_counts}", file=f)


""" Section: Visualize some example images """
import matplotlib.pyplot as plt

# Define width and height (easier to adjust)
width = 15
height = 4

# Create figure and subplots
plt.figure(figsize=(width, height)) 
count = 0
for example in dataset.take(5):   # Take 5 examples
  plt.subplot(1, 5, count + 1)
  image, label = example  
  plt.imshow(image[..., :3])  # Display only RGB channels
  plt.title(f"Label: {class_names[label.numpy()]}")
  #plt.show()
  count+=1
plt.savefig('example_figures.png')

# Additional exploration (optional)
# - Calculate statistics on image pixel values
# - Plot histograms of pixel values