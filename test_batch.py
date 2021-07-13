# %%
from matplotlib import pyplot as plt
from PIL import Image

import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import sys

import scale_and_slide as sas

# %%
# classes dict
classes = {
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
    # 10: 'background'
}

# %%
# load model
new_model = tf.keras.models.load_model('saved_model/my_model')
new_model.summary()

# %%

test_subdir = '5'
imgs_dir = f'{os.getcwd()}/debugging/train/{test_subdir}/'
window_size = (50, 50)
stride = 40
input_shape = (256, 256)

# %%
    
crops = []
resized_crops = []
png_files = os.listdir(imgs_dir)

for i, filename in enumerate(png_files):
    print(filename)
    # remove breakpoint to contiue through all pics in directory
    if i == 1: 
        break

    current_im = Image.open(imgs_dir + filename)
    chunks = sas.get_image_chunks(current_im, window_size, stride)

    # chucks is a nest list of images - combine sublists into
    # on big list
    for sublist in chunks:
        crops += sublist

    # print statement for debugging
    for i, x in enumerate(crops):
        if i == 0:
            print(f'shape after get_chunks: {x.shape}')
            break
    
    # resize crops to fit model input shape
    for i, x in enumerate(crops):
        if i == 0: # for debugging
            print(f'resizing from size: {x.shape}')

        resized = cv.resize(x, input_shape)
        resized_crops.append(resized)

        if i == 0: # for debugging
            print(f'shape of resized image: {resized.shape}')
    # for debugging
    print(f'total number of crops: {len(resized_crops)}')

# convert list to numpy array
image_array = np.array(resized_crops)
print(f'final shape of images inside of array: {image_array.shape} \n')

# ds = tf.data.Dataset.from_tensor_slices(image_array)
predictions = new_model.predict_on_batch(image_array[0:100])

print(f'images are coming from dir {test_subdir}: '
      f'{classes[int(test_subdir)]}, which is the expected output of '
       'all predictions in this run of the model\n')

paths_and_predictions = list(zip(png_files, predictions))

for i, pair in enumerate(paths_and_predictions):
    # remove breakpoint to continue through entire predictions set
    if i == 15:
        break
    # check score of ith image in dir
    score = tf.nn.softmax(pair[1])
    plt.figure()
    plt.title(f'model: this is a {classes[np.argmax(score)]}\n')
    plt.imshow(Image.open(imgs_dir + pair[0]))
    plt.show()
    print(f'score for test image {i}: {np.argmax(score)}')
    print(score)

# %%