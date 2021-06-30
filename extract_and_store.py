#%%
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json
import os

from PIL import Image

#%%
objects = ["bicycle", "motorcycle", "bus", "truck", "car", "train",
           "person", "traffic light", "stop sign", "fire hydrant"]
conversion_dict = {
    1:0,#bike
    3:1,#motorcycle
    5:2,#bus
    7:3,#truck
    2:4,#car
    6:5,#train
    0:6,#person
    9:7,#traffic light
    11:8,#stop sign
    10:9#fire hydrant
}

#%%
def string_to_array(str: str):
    array = str[1:-1].split(',')
    ret = []
    for value in array:
        value = int(value)
        #print(f'adding {value} to the array of type {type(value)}')
        ret.append(value)
    return ret


def change_label_array(v):
    v = string_to_array(v)
    for i in range(len(v)):
        v[i] = conversion_dict.get(v[i])
    return [x for x in v if x is not None]

def myprint(d):
    for k, v in d.items():
        if isinstance(v, dict):
            myprint(v)
        
        # save image as .jpeg
        elif k == 'image':
            im = Image.fromarray(v)
            global image_count
            # ************************************************************
            # MAKE SURE TO CHANGE DIRECTORY NAMEs TO THE LOCATION YOU WOULD
            # LOKE THE DATASET TO BE STORED
            # ************************************************************
            os.mkdir('/usr/nissi/coco_data_store/data/' + str(image_count))
            im.save('/usr/nissi/coco_data_store/data/' + str(image_count) + '/im_' + str(image_count) + '.jpeg')

        else:
            if isinstance(v, np.ndarray):
                v= str(v.tolist())
            elif isinstance(v, bytes):
                v= v.decode("utf-8")
            elif k == 'label':
                image = example['image']
                v = change_label_array(v)

            # write metadate to file in same directory as .jpeg
            # ************************************************************
            # MAKE SURE TO CHANGE DIRECTORY NAMEs TO THE LOCATION YOU WOULD
            # LOKE THE DATASET TO BE STORED
            # ************************************************************
            metadata = open("/usr/nissi/coco_data_store/data/" + str(image_count) + "/meta_" + str(image_count) + ".txt", "w")
            metadata.write(("{0} : {1}".format(k, v)))
            metadata.write("\n")
            metadata.close()
    
#%%
ds, info = tfds.load('coco', split='train', with_info=True)
ds = tfds.as_numpy(ds)
path_to_labels = "/home/ubuntu/tensorflow_datasets/coco/2014/1.1.0/objects-label.labels.txt"
with open(path_to_labels) as labels_file :
    word = labels_file.readlines()
#print(word)
lines = []
for obj in objects:
    #print(obj)
    for count, value in enumerate(word):
        if obj in value:
            lines.append(count)


#%%
# make sure the base directory exists before saving files in it
# ************************************************************
# MAKE SURE TO CHANGE DIRECTORY NAMEs TO THE LOCATION YOU WOULD
# LOKE THE DATASET TO BE STORED
# ************************************************************
os.makedirs("/usr/nissi/coco_data_store/data")
x = 0
image_count = 0
# remove breakpoint to continue though whole dataset
for example in ds:  # (image[], labels[], objects[], bbox[])
    # if x == 15:
    #     break
    image = example['image']
    labels = example['objects']['label']
    bboxes = example['objects']['bbox'] 

    #print(labels)
    # print(bboxes)  # [y_min, x_min, y_max, x_max]
    if (set(lines) & set(labels)):
        myprint(example)      
        image_count += 1
    x += 1


# %%