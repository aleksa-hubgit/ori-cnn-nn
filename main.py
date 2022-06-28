import os
import tensorflow as tf
import imghdr
import numpy as np
from keras import layers
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import Sequential

# loading the data

labels = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
          "Meat", "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-fruit"]
training_data = tf.keras.utils.image_dataset_from_directory('data\\training', image_size=(64, 64))
evaluation_data = tf.keras.utils.image_dataset_from_directory('data\\evaluation', image_size=(64, 64))
validation_data = tf.keras.utils.image_dataset_from_directory('data\\validation', image_size=(64, 64))

# scaling the data to an interval between 0 and 1
training_data = training_data.map(lambda x, y: (x / 255, y))
evaluation_data = evaluation_data.map(lambda x, y: (x / 255, y))
validation_data = validation_data.map(lambda x, y: (x / 255, y))

AUTOTUNE = tf.data.AUTOTUNE

training_data = training_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_data = validation_data.cache().prefetch(buffer_size=AUTOTUNE)
# model
model = Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(11, activation='sigmoid')
])


model.compile('adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(training_data, epochs=10, validation_data=validation_data)
model.summary()

