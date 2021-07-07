#%%

import matplotlib.pyplot as plt
import numpy as np
import math

from PIL import Image

#%%

def simple_display_image(img: Image.Image):
    plt.figure()
    plt.imshow(img)
    plt.show()

#%%

def display_crops(img: Image.Image, crops: list):
    rows = 1 + math.ceil((img.height - window_dimensions[1]) / stride)
    columns = 1 + math.ceil((img.width - window_dimensions[0]) / stride)
    figsize = (10 * columns, 10 * rows)

    print('num rows: ' + str(rows))
    print('num columns: ' + str(columns))

    fig = plt.figure(figsize=figsize)

    for i in range(0, columns * rows):
        try:
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(crops[i]) 
        except IndexError as ie:
            print('index error at index: ' + str(i))
            raise ie

    plt.show()

#%%

def get_scaled_images(img: Image.Image, count: int, increment: float):
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

def sliding_window(image: Image.Image, window_dim: tuple, stride: int):
    """iteratively slides a box across an image, cropping the image
    inside of the bounding box after each iteration. The crops are
    returned inside of an array
    Args:
        image (np.array): the image from which the crops wil be made
        window_dim (tuple): the dimensions of the sliding window, where
                            the 0th index holds the width and 1st slot
                            holds the width
        stride (int): how much to slide the window after 
                               each iteration
    Returns:
        list of PIL.Image.Image: the cropped images
    """
    print('sliding window over im size: ' + str(image.size))
    pictures = []

    tally = 0

    for y in range(0, image.height - stride, stride):
        for x in range(0, image.width - stride, stride):
            tally += 1
            # set crop bounds
            left_bound = x
            right_bound = left_bound + window_dim[0]
            upper_bound = y
            lower_bound = upper_bound + window_dim[1]

            # adjust bounds in cases of overflow
            if img.height < upper_bound + window_dim[1]:
                upper_bound = img.height - window_dim[1]
                lower_bound = img.height

            if img.width < left_bound + window_dim[0]:
                left_bound = img.width - window_dim[0]
                right_bound = img.width
            
            crop_bounds = [left_bound, upper_bound, right_bound, lower_bound]

            crop = img.crop(crop_bounds)
            pict = [crop, crop_bounds]
            pictures.append(pict)

    print('total sliding crops: ' + str(tally))

    return [im for [im, dims] in pictures]

#%%

def get_image_chunks(img: Image.Image, window_dim: tuple, stride: int, 
                     num_rescales: int, rescale_increment: float):
    """Generates sub-images by resizing the image and running a sliding
    window across each of the resized images
    Args:
        img (np.ndarray): image to process
        window_dim (tuple): dimensions of the sliding window
        stride (int): amount to slide the window after 
                               each iteration
        num_rescales (int): number of rescaled images to make
        rescale_increment (float): fraction by which to increase image 
                                   after each resize
    Returns:
        list: a nested list of lists of PIL.Image.Image of cropped images. Each
        resized image is stored inside of its own list
    """

    # first task - rescale the image into multiple sizes
    scaled_images = get_scaled_images(img=img, count=num_rescales, 
                                      increment=rescale_increment)

    # next - slide a window and crop sub-images from each of the
    # rescaled images
    crops = []
    
    for im in scaled_images:
        print('cropping im size:')
        print(im.size)
        crops.append(sliding_window(im, window_dim, stride))
        

    return crops

# %%

# ************************************
# SAMPLE TEST
# ************************************

img = Image.open('test.jpg')
window_dimensions = (200, 200)
stride = 100
num_rescales = 3
rescale_increment = .5

crops = get_image_chunks(img, window_dimensions, stride,
                         num_rescales, rescale_increment)

rescaled_images = get_scaled_images(img, num_rescales, rescale_increment)
simple_display_image(img)


# display_crops(rescaled_images[0], crops[0])
# display_crops(rescaled_images[1], crops[1])
display_crops(rescaled_images[2], crops[2])

# %%
