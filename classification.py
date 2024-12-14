import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from preprocessing import train_ds, test_ds

""" Section: Build the Model """
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 classes for EuroSAT
])

""" Section: Compile the Model """
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

""" Section: Train the Model """
model.fit(train_ds, epochs=10, validation_data=test_ds)

""" Section: Save the Model """
try:
    model.save('eurosat_cnn_model.keras')
    print('Model saved successfully!')
except Exception as e:
    print(f"Error saving model: {e}")

""" Section: Briefly Evaluate the Model """
test_loss, test_acc = model.evaluate(test_ds)
print('Test accuracy:', test_acc)