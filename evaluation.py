import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocessing import train_ds, test_ds

""" Section: Load the Saved Model """
model = load_model('eurosat_cnn_model.keras')  # Replace with the filename you used in classification.py

""" Section: Evaluate the Model """
test_loss, test_acc = model.evaluate(test_ds)
with open('evaluation_output.txt', 'w') as f:  # Open file for writing
    print('Test accuracy:', test_acc, file=f)