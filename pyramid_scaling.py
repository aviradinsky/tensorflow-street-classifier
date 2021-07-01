#%%
import matplotlib.pyplot as plt
import numpy as np
import math

from PIL import Image

#%%
def get_resized(count, increment, img) :
    """returns a list of multiple versions of one image, each of a differnt size

    Args:
        count (int): number of images to return (will be length of the array that is returned)
        increment (float): percentage increase from one size to the next
        img (numpy.array): image to resize
    """
# number of images smaller than img, will be determined by math.floor(num/2)
#     ex-odd: math.floor(3/2) = 1  ---  num == 3
#     ex-even: math.floor(4/2) = 2  ---  num == 4
# number of images larger than img, will be determined by math.ceiling(num/2) - 1
#     ex-odd: math.ceiling(3/2) - 1 = 1  ---  num == 3
#     ex-even: math.ceiling(4/2) - 1 = 1  ---  num == 4

    # increment is less than one; adding it to one will cause an increase in image resizing
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

    # apppend the original image before adding larger images
    imgs.append(img)

    # append larger images
    for i in reversed(range(num_larger_images)) :
        resize_amount = increment ** (num_larger_images - i)
        width = img_dims[0] * resize_amount
        length = img_dims[1] * resize_amount
        new_dims = (int(width), int(length))
        imgs.append(img.resize(new_dims))
    
    return imgs
# %%

img = Image.open('test.jpg')
num_images = 5
percentage_increase = .4
images = get_resized(num_images, percentage_increase, img)

## uncomment to save the images, you'll be able to see the size difference
# for i in range(len(images)) :
#     images[i].save(str(i) + '.png')

print('number of images: ' + str(num_images))
print('percentage increase: ' + str(percentage_increase))
print('original size of image: (' + str(img.size[0]) + ',' + str(img.size[1]) + ')')
print('new image sizes (from smallest to largest): \n')

for i in range(num_images) :
    print(images[i].size)

print()

# # display all the images (you can't see the size difference in pyplot, but image size is printed above each image)
# fig = plt.figure(figsize=(10,10))
# rows = 3
# columns = 3

# for i in range(num_images) :
#     fig.add_subplot(rows, columns, i + 1)
#     plt.imshow(images[i])
#     plt.axis('off')
#     plt.title(str(images[i].size))

# fig.show()

# %%
