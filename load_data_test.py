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

root = f'{os.getcwd()}/debugging/train'

# for directory in os.listdir(root):
#     print(directory)
#     print(len(os.listdir(root + '/' + directory)))

# %%
# check if directories exist
# if directories exist it assumes that all data is downloaded and the code begins running the model stuff
no_images = True
root = f'{os.getcwd()}/debugging'
for i in range(len(chosen_labels) + 1):
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

plt.figure(figsize=(10,10))

for ims, labs in train_data.take(1):
    for i in range(10):
            # object[int(labels[i])]
            ax = plt.subplot(6, 6, i + 1)
            plt.imshow(ims[i].numpy().astype("uint8"))
            plt.title(int(labs[i]))
            plt.axis("off")

print(type(ims))

# for img in ims:
#     print(img[0].shape)
print('print')

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
    # 10:  0
    }
for x in range(1,17): 
    print(x)
    plt.figure(figsize=(10, 10))
    for images, labels in train_data.take(1):
        for i in range(32):
            # object[int(labels[i])]
            ax = plt.subplot(6, 6, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
print(object)

# %%
