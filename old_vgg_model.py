# %%
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.core import Dropout
import download_data
import os
from params import chosen_labels, image_size, model_dir, new_labels, data_dir
# %%
# %%
"""
this loads all of the data from the tfds into folders
"""
download_data.main()
# %%
directory = f'{os.getcwd()}/{datadir}'

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=f'{directory}/train',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    image_size=image_size[0:2],
    shuffle=True,
    subset='training',
    validation_split=0.2,
    seed=6
)
validate_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=f'{directory}/train',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    image_size=image_size[0:2],
    shuffle=True,
    subset='validation',
    validation_split=0.2,
    seed=6
)
# %%
num_classes = len(chosen_labels) + 1 # for the background
data_augmentation = Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal')
])
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=image_size),
    data_augmentation,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes)
])
# %%
model.summary()
# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])
# %%
epochs = 15
history = model.fit(
    train_data,
    validation_data=validate_data,
    epochs=epochs
)
model.save(model_dir)
import confusionMatrix
# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
#%%
#%%
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=f'{directory}/test',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    image_size=image_size[:2],
    shuffle=True,
    seed=7
)
test_loss, test_acc = model.evaluate(test_data, verbose=2)
print(f'test_acc = {test_acc}')
#%%
