#%%
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dropout
# %%
DATASET_PATH = f'{os.getcwd()}/instances/'
print(DATASET_PATH)

# %%
train_data=tf.keras.preprocessing.image_dataset_from_directory(
    directory = DATASET_PATH,
    labels = 'inferred',
    label_mode='int',
    color_mode='rgb',
    validation_split=0.1,
    subset='training',
    seed=613
)

validate_data=tf.keras.preprocessing.image_dataset_from_directory(
    directory = DATASET_PATH,
    labels = 'inferred',
    label_mode='int',
    color_mode='rgb',
    validation_split=0.1,
    subset='validation',
    seed=613
)
# %%
num_classes = 10

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(256, 256, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dropout(0.2),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# %%
epochs=1
history = model.fit(
  train_data,
  validation_data=validate_data,
  epochs=epochs
)
# %%

