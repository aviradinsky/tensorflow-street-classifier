# %%
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
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
root = f'{os.getcwd()}/cropped_images'
for i in range(10):
    if os.path.exists(f'{root}/{i}'):
        continue
    else:
        os.makedirs(f'{root}/{i}')
#%%
data = tfds.load('coco').get('train')
# %%
i = 1
global count
count = 0
for sample in data:
    if i > 105:
        break
    else:
        i += 1

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

