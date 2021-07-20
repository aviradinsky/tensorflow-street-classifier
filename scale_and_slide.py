#%%

import matplotlib.pyplot as plt
import numpy as np
import math
import sys

from PIL import Image

#%%

def simple_display_image(img: Image.Image):
    """displays single image in pyplot

    Args:
        img (PIL.Image.Image): image to display
    """

    plt.imshow(img)
    plt.show()

#%%

def display_crops(img: Image.Image, crops: list, 
                  window_dimensions: tuple, stride: int):
    """displays gropu of image crops in a nicely formated rectangle

    Args:
        img (PIL.Image.Image): image whose crops are being displayed
        crops (list): list of image crops as numpy.arrays
        window_dimensions (tuple): dimensions of the cropping window
        stride (int): the stride
    """
    # print('BEGIN PYPLOT CROPS DISPLAY')
    rows = 1 + math.ceil((img.height - window_dimensions[1]) / stride)
    columns = 1 + math.ceil((img.width - window_dimensions[0]) / stride)
    figsize = (10 * columns, 10 * rows)

    print('rows: ' + str(rows))
    print('columns: ' + str(columns))

    fig = plt.figure(figsize=figsize)

    for i in range(0, columns * rows):
        if i >= len(crops):
            break
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(crops[i])
        plt.axis('off')

    plt.show()

    # print('END PYPLOT CROPS DISPLAY \n')

#%%

def get_scaled_images(img: Image.Image, count=3, increment=.5):
    """returns a list of multiple versions of one image, each of a 
    differnt size
    Args:
        img (PIL.Image.Image): image to resize
        count (int): number of images to return (will be length of the 
        array that is returned)
        increment (float): percentage increase from one size to the next
    Returns:
        list of PIL.Image.Image: the resized images
    """
    
    # print('START IMAGE RESCALING')
    imgs = []

    # add images from smallest to largest
    for i in range(count - 1) :
        resize_amount = increment ** (count - 1 - i)
        new_dims = (
                    int(img.width * resize_amount), 
                    int(img.height * resize_amount)
                    )
        # new_dims = (int(width), int(length))
        imgs.append(img.resize(new_dims))
        # print('image size: ', new_dims)

    # apppend the original image before adding larger images
    imgs.append(img)
    
    # print('original image size: ', tuple(img.size))
    # print('END IMAGE RESCALING \n')
    return imgs

#%%

def sliding_window(image: Image.Image, window_dim: tuple, stride: int):
    """iteratively slides a box across an image, cropping the image
    inside of the bounding box after each iteration. The crops are
    returned inside of an array
    Args:
        image (PIL.Image.Image): the image from which the crops wil be made
        window_dim (tuple): the dimensions of the sliding window, where
                            the 0th index holds the width and 1st slot
                            holds the width
        stride (int): how much to slide the window after 
                               each iteration

    Raises:
        ValueError: if window_dim is larger than image to crop

    Returns:
        list of (PIL.Image.Image, tuple) tuples: a list of tuples of 
        length 2, where the crop is stored at position [0] and its
        bounds are stored in a four-tuple at position [1]. The order
        of the tuple is as follows:
                (left_bound, upper_bound, right_bound, lower_bound)
    """

    # print('START WINDOW SLIDING')
    # print('cropping im size:' + str(image.size))

    if window_dim[0] > image.width or window_dim[1] > image.height:
        print(f'Crop dimenstions of {window_dim} were bigger than image'
                'size of {image.size}. Aborting crop.')
        return []

    pictures = []

    tally = 0

    for y in range(0, image.height, stride):
        for x in range(0, image.width, stride):
            tally += 1
            # set crop bounds
            left_bound = x
            right_bound = left_bound + window_dim[0]
            upper_bound = y
            lower_bound = upper_bound + window_dim[1]

            # adjust bounds in cases of overflow
            if image.height < lower_bound:
                upper_bound = image.height - window_dim[1]
                lower_bound = image.height

            if image.width < right_bound:
                left_bound = image.width - window_dim[0]
                right_bound = image.width
            
            crop_bounds = (left_bound, upper_bound, right_bound, lower_bound)

            crop = image.crop(crop_bounds)
            pict = (crop, crop_bounds)
            pictures.append(pict)

            if right_bound == image.width:
                break

    # print('total sliding crops: ' + str(tally))
    # print('END WINDOW SLIDING \n')

    return pictures

#%%

def get_image_chunks(img: Image.Image, window_dim: tuple, stride: int, 
                     num_rescales=3, rescale_increment=.5, 
                     display_imgs=False):
    """Generates sub-images by resizing the image and running a sliding
    window across each of the resized images
    Args:
        img (PIL.Image.Image): image to process
        window_dim (tuple): dimensions of the sliding window
        stride (int): amount to slide the window after 
                               each iteration
        num_rescales (int, optional): number of rescaled images to make
        rescale_increment (float, optional): fraction by which to 
                                 increase image after each resize
    Returns:
        list: a list tuples. Each crop is represented by a tuple of 
        length 2, where the array of the crop is stored at position
        [0] and its bbox in the original image is stored at 
        position [1].  The order of the bbox is:
                                (left, top, right, bottom)
    """

    # first task - rescale the image into multiple sizes
    scaled_images = get_scaled_images(img=img, count=num_rescales, 
                                      increment=rescale_increment)

    # next - slide a window and crop sub-images from each of the
    # rescaled images
    crops = []
    
    # convert images to numpy arrays
    for i, im in enumerate(scaled_images):
        scale = rescale_increment ** (len(scaled_images) - 1 - i)
        crops_set = sliding_window(im, window_dim, stride)
        crops_as_arrays = [(np.array(x), (int(y[0] * scale), int(y[1] * scale), 
                                          int(y[2] * scale), int(y[3] * scale)
                                          )
                            ) 
                            for x, y in crops_set
                          ]
        crops.append(crops_as_arrays)

    # for crop_list in crops:
    #     print('\n*********starting new list*********\n')
    #     for tup in crop_list:
    #         print((tup[1][2] - tup[1][0], tup[1][3] - tup[1][1]))

    if display_imgs:

        for i in range(num_rescales):
            # extract only the images from (crop, location) tuples
            crop_set = [x for x, y in crops[i]]
            display_crops(scaled_images[i], crop_set, window_dim, stride)

    simple_list = []

    for a_set in crops:
        simple_list += a_set

    # for tup in simple_list:
    #     print(tup[1])
        # print((tup[1][0] + tup[1][2], tup[1][1] + tup[1][3]))
    # print(type(simple_list[0][0]), type(simple_list[0][1]))
    # sys.exit()

    return simple_list

# %%

# ************************************
# SAMPLE TEST
# ************************************
def test():
    img = Image.open('cropped_images/train/1/173.png')
    simple_display_image(img)

    window_dimensions = (75, 100)
    stride = 50
    num_rescales = 3
    rescale_increment = .5

    # crops then displays images
    crops = get_image_chunks(img, window_dimensions, stride,
                             num_rescales, rescale_increment,
                             display_imgs=True
                            )

# %%

if __name__ == '__main__':
    test()

# %%
