#%%
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten
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
    seed=6
)

validate_data=tf.keras.preprocessing.image_dataset_from_directory(
    directory = DATASET_PATH,
    labels = 'inferred',
    label_mode='int',
    color_mode='rgb',
    validation_split=0.1,
    subset='validation',
    seed=6
)
# %%
num_classes = 10

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(256, 256, 3)),
  layers.Flatten(input_shape=(256,256,3)),
  layers.Dense(256,activation='relu'),
  layers.Dense(256,activation='relu'),
  layers.Dense(256,activation='relu'),
  layers.Dense(256,activation='relu'),
  layers.Dense(256,activation='relu'),
  layers.Dense(256,activation='relu'),
  layers.Dense(256,activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# %%
i = 1
for j in train_data:
  print(type(j[0]))
  if i < 4407:
    i += 1
    print(i)
    continue
  else:
    plt.figure()
    plt.imshow(j)
    plt.colorbar()
    plt.grid(False)
    plt.show()
    break
# %%
epochs=1
history = model.fit(
  train_data,
  validation_data=validate_data,
  epochs=epochs
)
# %%

