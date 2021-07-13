# %%
from matplotlib import pyplot as plt
from PIL import Image
from shutil import copyfile
import os
import glob
import sys

# %%
path = 'test'
# os.remove(f'{path}/5578.png')

# %%

cropped_images = os.getcwd() + '/cropped_images'
debugging = os.getcwd() + '/debugging'

train_count = 0
test_count = 0
val_count = 0

num_large_images = 0

# loops through train, test, and val
for data_subset_dir in os.listdir(cropped_images):
    data_subset_dir = '/' + data_subset_dir
    # loops through number dirs 0,1,2,3, etc
    for number_dir in os.listdir(cropped_images + data_subset_dir):
        
        # skip background class images
        if number_dir == '10':
            print(f'skipping background images in ' 
                  f'{cropped_images}/{number_dir}')
            continue

        number_dir = '/' + number_dir

        # loops through individual images
        for image in os.listdir(cropped_images + data_subset_dir 
                                + number_dir):
            image = '/' + image

            if data_subset_dir == '/train' and num_large_images == 593:
                train_count += num_large_images
                num_large_images = 0
                break
            elif data_subset_dir == '/test' and num_large_images == 78:
                test_count += num_large_images
                num_large_images = 0
                break
            elif data_subset_dir == '/val' and num_large_images == 73:
                val_count += num_large_images
                num_large_images = 0
                break

            pict_path = cropped_images + data_subset_dir\
                                + number_dir + '/' + image
            pict_destination = debugging + data_subset_dir\
                                + number_dir + '/' + image
            pict = Image.open(pict_path)
            if pict.height >= 50 and pict.width >= 50:
                copyfile(pict_path, pict_destination)
                num_large_images += 1

print(f'train count: {train_count}')
print(f'test count : {test_count}')
print(f'val count  : {val_count}')

# %%

test  = '/usr/nissi/tensorflow-street-classifier/debugging/test'
train = '/usr/nissi/tensorflow-street-classifier/debugging/train'
val   = '/usr/nissi/tensorflow-street-classifier/debugging/val'

def display_dir_imgs(path_to_dir: str):
    # loop through numbers
    for number in os.listdir(path_to_dir):
        for image in os.listdir(path_to_dir + '/' + number):
            img = Image.open(path_to_dir  + '/' + number + '/' + image)
            plt.figure()
            plt.imshow(img)
            plt.title(f'{path_to_dir}/{number}/{image}\nsize:{img.size}')
            plt.show()

# display_dir_imgs(test)
# display_dir_imgs(train)
# display_dir_imgs(val)

# %%

def empty_dirs(path_to_dir):
    # loop through numbers
    for number in os.listdir(path_to_dir):
        for image in os.listdir(path_to_dir + '/' + number):
            os.remove(path_to_dir + '/' + number + '/' + image)

test  = '/usr/nissi/tensorflow-street-classifier/debugging/test'
train = '/usr/nissi/tensorflow-street-classifier/debugging/train'
val   = '/usr/nissi/tensorflow-street-classifier/debugging/val'

# empty_dirs(test)
# empty_dirs(train)
# empty_dirs(val)

