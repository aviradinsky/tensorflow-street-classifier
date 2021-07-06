#%%

import matplotlib.pyplot as plt
import numpy as np
import math

from PIL import Image

#%%

def get_scaled_images(img: np.array, count: int, increment: float):
    """returns a list of multiple versions of one image, each of a 
    differnt size

    Args:
        count (int): number of images to return (will be length of the 
        array that is returned)
        increment (float): percentage increase from one size to the next
        img (numpy.array): image to resize
    
    Returns:
        list of PIL.Image.Image: the resized images
    """
# number of images smaller than img will be determined 
# by math.floor(num/2)
#     ex-odd: math.floor(3/2) = 1  ---  num == 3
#     ex-even: math.floor(4/2) = 2  ---  num == 4
# number of images larger than img will be determined 
# by math.ceiling(num/2) - 1
#     ex-odd: math.ceiling(3/2) - 1 = 1  ---  num == 3
#     ex-even: math.ceiling(4/2) - 1 = 1  ---  num == 4

    # increment is less than one; adding it to one will cause an 
    # increase in image resizing
    print('START IMAGE RESCALING')
    increment += 1
    num_smaller_images = math.floor(count/2)
    num_larger_images = math.ceil(count/2) - 1
    img_dims = img.size
    imgs = []

    # add smaller images to list first 
    for i in range(num_smaller_images) :
        resize_amount = 1 / (increment ** (num_smaller_images - i))
        width = img_dims[0] * resize_amount
        length = img_dims[1] * resize_amount
        new_dims = (int(width), int(length))
        imgs.append(img.resize(new_dims))
        print('image size: ', new_dims)

    # apppend the original image before adding larger images
    imgs.append(img)
    print('original image size: ', tuple(img.size))

    # append larger images
    for i in reversed(range(num_larger_images)) :
        resize_amount = increment ** (num_larger_images - i)
        width = img_dims[0] * resize_amount
        length = img_dims[1] * resize_amount
        new_dims = (int(width), int(length))
        imgs.append(img.resize(new_dims))
        print('image size: ', new_dims)

    print('END IMAGE RESCALING')

    return imgs

#%%

#crops the images into 200 x 200 smaller images and saves them along 
# with its location on the original picture in the pictures list
def sliding_window(image: np.array, window_dim: tuple, slide_increment: int):
    """iteratively slides a box across an image, cropping the image
    inside of the bounding box after each iteration. The crops are
    returned inside of an array

    Args:
        image (np.array): the image from which the crops wil be made
        window_dim (tuple): the dimensions of the sliding window, where
                            the 0th index holds the width and 1st slot
                            holds the width
        slide_increment (int): how much to slide the window after 
                               each iteration

    Returns:
        list of PIL.Image.Image: the cropped images
    """

    pictures = []

    for y in range(0, img.height, slide_increment):
        for x in range(0, img.width, slide_increment):
            new_x = x + window_dim[0]
            new_y = y + window_dim[1]
            crop_dims = [x, y, new_x, new_y]
            crop = img.crop(crop_dims)
            pict = [crop, crop_dims]
            pictures.append(pict)

    return [im for [im, dims] in pictures]

#%%

def get_image_chunks(img: np.ndarray, window_dim: tuple, slide_increment: int, 
                     num_rescales: int, rescale_increment: float):
    """Generates sub-images by resizing the image and running a sliding
    window across each of the resized images

    Args:
        img (np.ndarray): image to process
        window_dim (tuple): dimensions of the sliding window
        slide_increment (int): amount to slide the window after 
                               each iteration
        num_rescales (int): number of rescaled images to make
        rescale_increment (float): fraction by which to increase image 
                                   after each resize

    Returns:
        list: a list of PIL.Image.Image of cropped images
    """

    # first task - rescale the image into multiple sizes
    scaled_images = get_scaled_images(img = img, count=num_rescales, 
                                      increment=rescale_increment)
    print(type(scaled_images[0]))

    # next - slide a window and crop sub-images from each of the
    # rescaled images
    crops = []
    
    for image in scaled_images:
        crops += sliding_window(image, window_dim, slide_increment)

    return crops

# %%

# ************************************
# SAMPLE TEST
# ************************************

img = Image.open('test.jpg')
window_dimensions = (200, 200)
slide_increment = 100
num_rescales = 2
rescale_increment = .5

crops = get_image_chunks(img, window_dimensions, slide_increment,
                         num_rescales, rescale_increment)

plt.figure()
plt.imshow(img)
plt.show()

#  displays all of the images in one pyplot
fig = plt.figure(figsize=(60,60))

columns = (img.height // slide_increment) + 1
rows = (img.width // slide_increment) + 1

for i in range(1, columns * rows + 1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(crops[i - 1])
    
plt.show()

print('crops size: ', crops[0].size)
print('number of image crops is:', len(crops))
print(type(crops[0]))


#%%