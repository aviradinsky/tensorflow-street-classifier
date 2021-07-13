# %%
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten
from numpy import random
import numpy as np
# %%
# loading the data
chosen_labels = [
    0,   # 'bike',
    1,   # 'motorcycle',
    2,   # 'bus',
    3,   # 'truck',
    4,   # 'car',
    5,   # 'train',
    6,   # 'person',
    7,   # 'traffic light',
    8,   # 'stop sign',
    9,   # 'fire hydrant',
    # 10  # 'background'
]

# %%

root = f'{os.getcwd()}/debugging/train'

for directory in os.listdir(root):
    print(directory)
    print(len(os.listdir(root + '/' + directory)))

# %%
# check if directories exist
# if directories exist it assumes that all data is downloaded and the code begins running the model stuff
no_images = True
root = f'{os.getcwd()}/debugging'
# for i in range(len(chosen_labels) + 1):
for i in range(len(chosen_labels)):
    if os.path.exists(f'{root}/test') and os.path.exists(f'{root}/val') and os.path.exists(f'{root}/train'):
        no_images = False
        break
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
        # adding all background to 10
        if len(set(all_labels) & set(chosen_labels)) == 0:
                file_name = f'{count}'
                count += 1
                file_location = f'{root}/10'
                tf.keras.preprocessing.image.save_img(
                    f'{file_location}/{file_name}.png', image.numpy())
                if count % 1000 == 0:
                    print(
                        f'Number of images placed into directory structure: {count}')
                continue


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
    for i in range(11):
        print(i)
        try:
            os.makedirs('cropped_images/train/'+i)
            os.makedirs('cropped_images/val/'+i)
            os.makedirs('cropped_images/test/'+i)
        except IOError as e:
            print("already dir")
        
        arr= os.listdir('cropped_images/'+i)
        arr.sort()  # make sure that the filenames have a fixed order before shuffling
        random.seed(230)
        random.shuffle(arr)
        train = int(.8*len(arr))
        val = int(.9*len(arr))
        test = len(arr)
        train_data= arr[0:train]
        val_data= arr[train:val]
        test_data= arr[val:test]
        for trainName in train_data:
            os.rename('cropped_images/'+i+'/'+ trainName,'cropped_images/train/'+i+'/'+ trainName)
        for valName in val_data:
            os.rename('cropped_images/'+i+'/'+ valName,'cropped_images/val/'+i+'/'+ valName)  
        for testName in test_data:
            os.rename('cropped_images/'+i+'/'+ testName,'cropped_images/test/'+i+'/'+ testName)   
        try:
            os.rmdir('cropped_images/'+i)
        except IOError as e:
            print("not empty dir")
# %%

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=root+"/train",
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    seed=6
)

validate_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=root +"/val",
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    seed=6
)

# %%

print(train_data)
#object = defaultdict(int)
object = {
    0:   0, # bike
    1:   0,
    2:   0, # airplane
    3:   0, # bus
    4:   0,
    5:   0, # car
    6:   0,
    7:   0, # person
    8:   0,
    9:   0, # stop sign
    10:  0
    }
# for x in range(1,17): 
#     print(x)
#     plt.figure(figsize=(10, 10))
#     for images, labels in train_data.take(1):
#         for i in range(32):
#             object[int(labels[i])]+=1
#             ax = plt.subplot(6, 6, i + 1)
#             plt.imshow(images[i].numpy().astype("uint8"))
#             plt.title(int(labels[i]))
#             plt.axis("off")
# print(object)


# %%

iter_ds = train_data.as_numpy_iterator()

for x, i in enumerate(train_data):
    # print(type(i), len(i), type(i[0]), type(i[1]))
    print(i[1])
    print(x)
    if x == 30:
        break


# %%
num_classes = len(chosen_labels) # for the background


# %%
print(num_classes)

# %%

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
model.summary()
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
# """
# Found 246943 files belonging to 10 classes.
# Using 222249 files for training.
# Found 246943 files belonging to 10 classes.
# Using 24694 files for validation.
# Epoch 1/10
# 6946/6946 [==============================] - 895s 107ms/step - loss: 0.6827 - accuracy: 0.7991 - val_loss: 0.4739 - val_accuracy: 0.8518
# Epoch 2/10
# 6946/6946 [==============================] - 638s 92ms/step - loss: 0.4361 - accuracy: 0.8623 - val_loss: 0.4173 - val_accuracy: 0.8685
# Epoch 3/10
# 6946/6946 [==============================] - 602s 87ms/step - loss: 0.3788 - accuracy: 0.8803 - val_loss: 0.4230 - val_accuracy: 0.8701
# Epoch 4/10
# 6946/6946 [==============================] - 602s 87ms/step - loss: 0.3427 - accuracy: 0.8901 - val_loss: 0.4407 - val_accuracy: 0.8681
# Epoch 5/10
# 6946/6946 [==============================] - 605s 87ms/step - loss: 0.3161 - accuracy: 0.8986 - val_loss: 0.4199 - val_accuracy: 0.8732
# Epoch 6/10
# 6946/6946 [==============================] - 601s 87ms/step - loss: 0.2962 - accuracy: 0.9032 - val_loss: 0.4259 - val_accuracy: 0.8750
# Epoch 7/10
# 6946/6946 [==============================] - 601s 87ms/step - loss: 0.2769 - accuracy: 0.9099 - val_loss: 0.4270 - val_accuracy: 0.8756
# Epoch 8/10
# 6946/6946 [==============================] - 601s 87ms/step - loss: 0.2628 - accuracy: 0.9140 - val_loss: 0.4574 - val_accuracy: 0.8770
# Epoch 9/10
# 6946/6946 [==============================] - 601s 86ms/step - loss: 0.2509 - accuracy: 0.9179 - val_loss: 0.4569 - val_accuracy: 0.8708
# Epoch 10/10
# 6946/6946 [==============================] - 606s 87ms/step - loss: 0.2351 - accuracy: 0.9230 - val_loss: 0.4458 - val_accuracy: 0.8749
# """

# %%
model.save('saved_model/my_model')
#%%
new_model = tf.keras.models.load_model('saved_model/my_model')
new_model.summary()

# %%
#testing a bike
object = {
    0:   'bike',
    1:   'motorcycle',
    2:   'bus',
    3:   'truck',
    4:   'car',
    5:   'train',
    6:   'person',
    7:   'traffic light',
    8:   'stop sign',
    9:   'fire hydrant',
    10: 'background'
    }
img = keras.preprocessing.image.load_img(
    str(os.getcwd()+ '/cropped_images/test/0/1279.png'), target_size=(256, 256)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(object[np.argmax(score)], 100 * np.max(score))
)
# %%
