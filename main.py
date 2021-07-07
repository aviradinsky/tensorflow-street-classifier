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
    1,  # bike
    3,  # motorcycle
    5,  # bus
    7,  # truck
    2,  # car
    6,  # train
    0,  # person
    9,  # traffic light
    11,  # stop sign
    10  # fire hydrant
]
# %%
# check if directories exist
# if directories exist it assumes that all data is downloaded and the code begins running the model stuff
no_images = True
root = f'{os.getcwd()}/cropped_images'
for i in range(10):
    if os.path.exists(f'{root}/{i}'):
        no_images = False
        continue
    else:
        os.makedirs(f'{root}/{i}')
# %%
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

                offset_height = int(top_line)
                offset_width = int(left_line)
                target_height = int(bottom_line - top_line)
                target_width = int(right_line - left_line)

                if target_height == 0 or target_width == 0 or top_line == 0 or right_line == 0:
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
                tf.keras.preprocessing.image.save_img(
                    f'{file_location}/{file_name}.png', cropped_image.numpy())
                if count % 1000 == 0:
                    print(
                        f'Number of images placed into directory structure: {count}')
                    break


# %%
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=root,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    validation_split=0.1,
    subset='training',
    seed=6
)

validate_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=root,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    validation_split=0.1,
    subset='validation',
    seed=6
)
# %%
num_classes = len(chosen_labels)
data_augmentation = Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal')
])

model = Sequential([

    layers.experimental.preprocessing.Rescaling(
        1./255, input_shape=(256, 256, 3)),
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
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])
# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])
# %%
epochs = 10
history = model.fit(
    train_data,
    validation_data=validate_data,
    epochs=epochs
)

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
"""
Found 246943 files belonging to 10 classes.
Using 222249 files for training.
Found 246943 files belonging to 10 classes.
Using 24694 files for validation.
Epoch 1/10
6946/6946 [==============================] - 895s 107ms/step - loss: 0.6827 - accuracy: 0.7991 - val_loss: 0.4739 - val_accuracy: 0.8518
Epoch 2/10
6946/6946 [==============================] - 638s 92ms/step - loss: 0.4361 - accuracy: 0.8623 - val_loss: 0.4173 - val_accuracy: 0.8685
Epoch 3/10
6946/6946 [==============================] - 602s 87ms/step - loss: 0.3788 - accuracy: 0.8803 - val_loss: 0.4230 - val_accuracy: 0.8701
Epoch 4/10
6946/6946 [==============================] - 602s 87ms/step - loss: 0.3427 - accuracy: 0.8901 - val_loss: 0.4407 - val_accuracy: 0.8681
Epoch 5/10
6946/6946 [==============================] - 605s 87ms/step - loss: 0.3161 - accuracy: 0.8986 - val_loss: 0.4199 - val_accuracy: 0.8732
Epoch 6/10
6946/6946 [==============================] - 601s 87ms/step - loss: 0.2962 - accuracy: 0.9032 - val_loss: 0.4259 - val_accuracy: 0.8750
Epoch 7/10
6946/6946 [==============================] - 601s 87ms/step - loss: 0.2769 - accuracy: 0.9099 - val_loss: 0.4270 - val_accuracy: 0.8756
Epoch 8/10
6946/6946 [==============================] - 601s 87ms/step - loss: 0.2628 - accuracy: 0.9140 - val_loss: 0.4574 - val_accuracy: 0.8770
Epoch 9/10
6946/6946 [==============================] - 601s 86ms/step - loss: 0.2509 - accuracy: 0.9179 - val_loss: 0.4569 - val_accuracy: 0.8708
Epoch 10/10
6946/6946 [==============================] - 606s 87ms/step - loss: 0.2351 - accuracy: 0.9230 - val_loss: 0.4458 - val_accuracy: 0.8749
"""
