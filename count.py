from matplotlib import pyplot as plt
from PIL import Image
from shutil import copyfile
import os
import glob
import sys

# %%
path = 'test'
os.remove(f'{path}/5578.png')

# %%

cropped_images = os.getcwd() + '/cropped_images'
# debugging = os.getcwd() + '/debugging'

num_large_images = 0

# loops through train, test, and val
for cropped_images_dir in os.listdir(cropped_images):
    cropped_images_dir = '/' + cropped_images_dir
    # loops through number dirs 0,1,2,3, etc
    for number_dir in os.listdir(cropped_images + cropped_images_dir):
        number_dir = '/' + number_dir
        # loops through individual images
        for image in os.listdir(cropped_images + cropped_images_dir + number_dir):
            pict_path = cropped_images + cropped_images_dir + number_dir + '/' + image
            # pict_destination = debugging + cropped_images_dir + number_dir + '/' + image
            pict = Image.open(pict_path)
            if pict.height >= 50 and pict.width >= 50:
                num_large_images += 1
                # files = glob.glob('debugging + cropped_images_dir + number_dir')
                # for f in files:
                #     print(f)
                #     sys.exit('exiting here')
                #     os.remove(f)
                # print(f'about to remove file at {pict_path}')
                # sys.exit('aborting file deltion')
                # os.remove(pict_path)
                # continue
        print(f'number of large images in {cropped_images_dir + number_dir} is: {num_large_images}')
        num_large_images = 0

