import json
import PIL
import tensorflow as tf
from matplotlib import pyplot as plt
batch_size = 1
img_height = 128
img_width = 128
import numpy as np
labels = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
          "Meat", "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-fruit"]

model = tf.keras.models.load_model('cnn_model.h5')
nmodel = tf.keras.models.load_model('nn_model.h5')

model.summary()
eval_ds_cnn = tf.keras.utils.image_dataset_from_directory(
    'data\\evaluation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
eval_ds_nn = tf.keras.utils.image_dataset_from_directory(
    'data\\evaluation',
    seed=123,
    image_size=(32, 32),
    batch_size=batch_size
)

history = json.load(open('NN_RESULTS', 'r'))
chistory = json.load(open('CNN_RESULTS', 'r'))

fig = plt.figure()
plt.plot(history['loss'], color='teal', label='NN loss')
plt.plot(history['val_loss'], color='orange', label='NN validation loss')
fig.suptitle('Losses', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(history['accuracy'], color='teal', label='NN accuracy')
plt.plot(history['val_accuracy'], color='orange', label='NN validation accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(chistory['loss'], color='blue', label='CNN loss')
plt.plot(chistory['val_loss'], color='red', label='CNN validation loss')
fig.suptitle('Losses', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(chistory['accuracy'], color='blue', label='CNN accuracy')
plt.plot(chistory['val_accuracy'], color='red', label='CNN validation accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# results = model.evaluate(eval_ds_cnn)
# print("CNN\nTest loss, test accuracy: ", results)
# results = nmodel.evaluate(eval_ds_nn)
# print("NN\nTest loss, test accuracy: ", results)

img = tf.keras.utils.load_img(
    'data/testjaje.jpg', target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)

img = tf.keras.utils.load_img(
    'data/testjaje.jpg', target_size=(32, 32)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

npredictions = nmodel.predict(img_array)

score = predictions[0]
nscore = npredictions[0]
print(labels, score)
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(labels[np.argmax(score)], 100 * np.max(score))
)
fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(labels, score, color='teal',
        width=0.4)

plt.title("CNN Prediction")
plt.show()

fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(labels, nscore, color='teal',
        width=0.4)

plt.title("NN Prediction")
plt.show()
