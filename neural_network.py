import json
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

batch_size = 32
img_height = 32
img_width = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    'data\\training',
    seed=123,
    image_size=(img_height, img_width),
    validation_split=0.2,
    subset="training",
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'data\\training',
    validation_split=0.2,
    seed=49,
    image_size=(img_height, img_width),
    subset="validation",
    batch_size=batch_size)


class_names = train_ds.class_names


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


num_classes = len(class_names)


model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Flatten(input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

epochs = 1000
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
model.summary()
json.dump(history.history, open('NN_RESULTS', 'w'))
model.save('nn_model.h5')
