# %%
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten
# %%
# loading the data
chosen_labels = [
    1, # bike
    3, # motorcycle
    5, # bus
    7, # truck
    2, # car
    6, # train
    0, # person
    9, # traffic light
    11, # stop sign
    10 # fire hydrant
]
# %%
# check if directories exist
no_images = True
root = f'{os.getcwd()}/cropped_images'
for i in range(10):
    if os.path.exists(f'{root}/{i}'):
        no_images = False
        continue
    else:
        os.makedirs(f'{root}/{i}')
#%%
data = tfds.load('coco').get('train')
# %%
i = 1
global count
count = 1
if no_images:
    for sample in data:
        image = sample.get('image')
        objects = sample.get('objects')
        all_labels = objects.get('label').numpy()
        for j, label in enumerate(all_labels):
            if label in chosen_labels:
                bbox = objects.get('bbox').numpy()[j]

                top_line = bbox[0]*image.shape[0]
                left_line = bbox[1]*image.shape[1]
                bottom_line = bbox[2]*image.shape[0]
                right_line = bbox[3]*image.shape[1]

                offset_height=int(top_line)
                offset_width=int(left_line)
                target_height=int(bottom_line - top_line)
                target_width=int(right_line - left_line)

                if target_height == 0 or target_width == 0:
                    continue

                cropped_image = tf.image.crop_to_bounding_box(image,
                    offset_height=offset_height,
                    offset_width=offset_width,
                    target_height=target_height,
                    target_width=target_width
                )

                file_name = f'{count}'
                count += 1
                file_location = f'{root}/{chosen_labels.index(label)}'
                tf.keras.preprocessing.image.save_img(f'{file_location}/{file_name}.png',cropped_image.numpy())
                print(f'Number of images placed into directory structure: {count}')


# %%
train_data=tf.keras.preprocessing.image_dataset_from_directory(
    directory = root,
    labels = 'inferred',
    label_mode='int',
    color_mode='rgb',
    validation_split=0.1,
    subset='training',
    seed=6
)

validate_data=tf.keras.preprocessing.image_dataset_from_directory(
    directory = root,
    labels = 'inferred',
    label_mode='int',
    color_mode='rgb',
    validation_split=0.1,
    subset='validation',
    seed=6
)
# %%
num_classes = len(chosen_labels)

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
epochs=1
history = model.fit(
  train_data,
  validation_data=validate_data,
  epochs=epochs
)