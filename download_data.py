# %%

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.backend import convert_inputs_if_ragged
from tensorflow.image import crop_to_bounding_box as crop_bbox
import tensorflow_datasets as tfds
import os
import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image
from params import data_dir, new_labels_list, all_possible_labels

# %%
def crop_tensor_by_nth_bbox(tensor, nth_bbox):
    """
    crop a tensor in the dataset by the nth tensor
    """
    bbox = tensor.get('objects').get('bbox').numpy()[nth_bbox]
    image = tensor.get('image')

    top = max(int(bbox[0] * image.shape[0]) - 15, 0)
    left = max(int(bbox[1] * image.shape[1]) - 15, 0)
    bottom = min(int(bbox[2] * image.shape[0]) + 15, image.shape[0])
    right = min(int(bbox[3] * image.shape[1]) + 15, image.shape[1])

    crops = []
    # get simple crop, no bbox shifting
    height = abs(bottom - top)
    width = abs(right - left)
    if height < 50 or width < 50: # don't crop too small bboxes
        return None
    crops.append(crop_bbox(image, top, left, height, width))
    # shift bbox up and crop
    new_top = int(max(top - (.3 * height), 0))
    if top - new_top >= 10: # only crop if significant shift is made
        crops.append(crop_bbox(image, new_top, left, height, width))
    # shift bbox down and crop
    new_bottom = int(min(bottom + (.3 * height), image.shape[0]))
    if new_bottom - bottom < 0:
        raise Exception('negative height value')
    if new_bottom - bottom >= 10:
        crops.append(crop_bbox(image, new_bottom - height, left, 
                               height, width))
    # shift bbox left and crop
    new_left = int(max(left - (.3 * width), 0))
    if left - new_left >= 10:
        crops.append(crop_bbox(image, top, new_left, height, width))
    # shift bbox right and crop
    new_right = int(min(right + (.3 * width), image.shape[1]))
    if new_right - right >= 10:
        crops.append(crop_bbox(image, top, new_right - width,
                               height, width))
    return crops

def get_img_crops(image: Image.Image, bbox: tuple):
    # tuple is in order: (left, top, right, bottom)
    # returns iterable of crops
    crops = []
    left, top, right, bottom = bbox
    left = max(left -  15, 0)
    top = max(top - 15, 0)
    right = min(right + 15, image.width)
    bottom = min(bottom + 15, image.height)
    height = abs(top - bottom)
    width = abs(right - left)
    if height < 50 or width < 50:
        return None
    # first add simple crop using original bbox dimensions
    crops.append(image.crop(bbox))
    # shift box up and crop
    new_top = int(max(top - .25 * height, 0))
    new_bottom = new_top + height
    if top - new_top >= 10: # only crop if there's a significant shift
        crops.append(image.crop((left, new_top, right, new_bottom)))
    # shift box down and crop
    new_bottom = int(min(bottom + .25 * height, image.height))
    new_top = new_bottom - height
    if new_bottom - bottom >= 10:
        crops.append(image.crop((left, new_top, right, new_bottom)))
    # shift box left and crop
    new_left = int(max(left - .25 * width, 0))
    new_right = new_left + width
    if left - new_left >= 10:
        crops.append(image.crop((new_left, top, new_right, bottom)))
    # chift box right and crop
    new_right = int(min(right + .25 * width, image.width))
    new_left = new_right - width
    if new_right - right >= 10:
        crops.append(image.crop((new_left, top, new_right, bottom)))
    return crops

# %%
def slice_into_4ths(image):
    x,y,z = image.shape

    x,y = int(x/2),int(y/2)

    top_left = tf.slice(image, [0, 0, 0], [x, y, 3])
    bottom_left = tf.slice(image, [x, 0, 0], [x, y, 3])
    top_right = tf.slice(image, [0, y, 0], [x, y, 3])
    bottom_right = tf.slice(image, [x, y, 0], [x, y, 3])

    return [
        top_right,
        top_left,
        bottom_left,
        bottom_right,
    ]

# %%

def sum_counts_without_background(counts: dict):
    total = 0
    for key in counts.keys():
        if key != 'background':
            total += counts[key]
    return total

# %%

def main(directory = f'{os.getcwd()}', chosen_labels_string = new_labels_list):
    all_labels = all_possible_labels
    all_labels.append('background')

    for label in chosen_labels_string:
        if label not in all_labels:
            raise Exception(f'No such label as {label}')

    directory = f'{directory}/{data_dir}'

    dont_need_to_continue = False

    for string in chosen_labels_string:
        if os.path.exists(f'{directory}/test/{string}/'):
            dont_need_to_continue = True
            break
        if os.path.exists(f'{directory}/train/{string}/'):
            dont_need_to_continue = True
            break

    if dont_need_to_continue:
        print('Directory structure for data detected\nno longer creating data')
        return

    count_of_labels_dict = {}

    for string in chosen_labels_string:
        if(string=="truck"):
            continue
        s = string.replace(' ', '_')
        count_of_labels_dict[s] = 0
        os.makedirs(f'{directory}/test/{s}/')
        os.makedirs(f'{directory}/train/{s}/')
    count_of_labels_dict['background'] = 0
    print(count_of_labels_dict)

    """
    writing the chosen labels into an array with their number value
    """
    chosen_labels_int = []
    for label in chosen_labels_string:
        if label in all_labels:
            chosen_labels_int.append(all_labels.index(label))
    print(chosen_labels_int)
    coco = tfds.load('coco')
    open_data = foz.load_zoo_dataset(
            "open-images-v6",
            split="train",
            label_types=["detections", "classifications"],
            classes = ["Bicycle", "Motorcycle", "Car", "Truck", "Train"],
            only_matching = True
    )
    open_data.persistent = True
    open_data.save()
    """
    all_data.keys() = 'test', 'test2015', 'train', 'validation'
    """
    valid_keys = ('train','validation')
    num_saved_imgs = 0
    imagecount = 0
    reached_test_mark = False
    used_image = False
    for key in valid_keys: # loop through train and validation sets
        for data in coco.get(key): # loop through each image in the set
            # to prevent unnecessary looping through entire dataset
            if sum_counts_without_background(count_of_labels_dict) >= 160000 * (len(chosen_labels_string) - 1):
                return
            images_to_send = []
            image = data.get('image')
            # get list of labels in the image
            number_form_labels = data.get('objects').get('label').numpy()
            # make lists of (crop,label) pairs in images_to_send
            for n, int_label in enumerate(number_form_labels):
                # filter out only labels we will use for model
                if int_label in chosen_labels_int:
                    crops = crop_tensor_by_nth_bbox(data, n)
                    if crops is None:
                        continue
                    label_list = [all_labels[int_label]] * len(crops)
                    zipped = list(zip(crops, label_list))
                    images_to_send.append(zipped)
            if len(images_to_send) == 0: # this img only conains background
                slices = slice_into_4ths(image)
                label_list = ['background'] * len(slices)
                zipped = list(zip(slices,label_list))
                images_to_send.append(zipped)
            # now save crops in proper directory
            for send in images_to_send:
                # set label
                label = send[0][1].replace(' ','_')
                if label == 'truck':
                    label = 'car'
                if count_of_labels_dict[label] > 160000:
                    continue
                if label == 'background':
                    imagecount += 4
                else:
                    imagecount += 1 # only count one for non-background images
                for i, crop in enumerate(send):
                    image = crop[0]
                    count_of_labels_dict[label] += .25 if label == 'background' else 1
                    full_label = None
                    if label == 'background':
                        full_label = f'test/{label}' if count_of_labels_dict[label] % 2.5 == 0 else f'train/{label}'
                    elif reached_test_mark:
                        full_label = f'test/{label}'
                        reached_test_mark = False
                    else:
                        full_label = f'train/{label}'

                    tf.keras.preprocessing.image.save_img(f'{directory}/{full_label}/{num_saved_imgs}.jpg',image)
                    num_saved_imgs += 1
                    
                    if num_saved_imgs % 1000 == 0:
                        print("number of used :" + str(num_saved_imgs) + " images used: " + str(imagecount))
                    # if img is test image, only save first crop
                    if full_label[0:4] == 'test' and full_label[5:9] is not 'back':
                        break
                    # check to see if this round pushed total over level
                    # where image should be sent to test set
                    if i == len(send) - 1 and label != 'background':
                        for j in range(len(send)):
                            if (count_of_labels_dict[label] - j) % 40 == 0:
                                reached_test_mark = True
            
    print("open images")
    for sample in open_data:
        # to prevent unnecessary looping through entire dataset
        if sum_counts_without_background(count_of_labels_dict) >= 160000 * (len(chosen_labels_string) - 1):
            return
        i = Image.open(sample["filepath"])
        height = i.height
        width = i.width
        images_to_send = []
        # get all bboxes
        boxes = sample["detections"]["detections"]
        # make lists of (crop,label) pairs in images_to_send
        for box in boxes:
            label = box['label'].lower().replace(" ", "_")
            if(label =="truck"):
                label = "car"
            if count_of_labels_dict.get(label,"no") == "no" or count_of_labels_dict[label] > 160000:
                continue
            cropSize = (box["bounding_box"][0] * width,
                        box["bounding_box"][1] * height,
                        box["bounding_box"][2] * width + box["bounding_box"][0] * width,
                        box["bounding_box"][3] * height + box["bounding_box"][1] * height)
            # crop = i.crop(cropSize)
            crops = get_img_crops(i, cropSize)
            if crops is None: # means bbox was too small (<50 pixels)
                continue
            label_list = [label] * len(crops)
            zipped = list(zip(crops, label_list))
            images_to_send.append(zipped)
        for send in images_to_send:
            label = send[0][1]
            if label not in new_labels_list:
                raise ValueError(f'unexpected bbox class: {label}')
            if label == 'background':
                raise ValueError('unexpected class: background')
            if count_of_labels_dict[label] > 160000:
                continue
            imagecount += 1
            # crop is tuple of (image, label)
            for i, crop in enumerate(send):
                image = crop[0]
                count_of_labels_dict[label] += 1
                full_label = None
                if reached_test_mark:
                    full_label = f'test/{label}'
                    reached_test_mark = False
                else:
                    full_label = f'train/{label}'
                crop[0].save(f'{directory}/{full_label}/{num_saved_imgs}.jpg')
                num_saved_imgs += 1

                if  num_saved_imgs % 1000 == 0:
                    print("number of used :" + str(num_saved_imgs) + " images used: " + str(imagecount))
                # if img is test image, only save first crop
                if full_label[0:4] == 'test':
                    break
                # check to see if this round pushed total over level
                # where image should be sent to test set
                if i == len(send) - 1 and label != 'background':
                    for j in range(len(send)):
                        if (count_of_labels_dict[label] - j) % 40 == 0:
                            reached_test_mark = True
                            
# %%

if __name__ == '__main__':
    main()

# %%
# code to empty directories without deleting them
# import os, shutil
# root = './newdatatest'
# for folder in os.listdir(root):
#     print(folder)
#     path = os.path.join(root, folder)
#     for clss in os.listdir(path):
#         file_path = os.path.join(path, clss)
#         for f in os.listdir(file_path):
#             final = os.path.join(file_path, f)
#             os.remove(final)

