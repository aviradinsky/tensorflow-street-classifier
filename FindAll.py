#%%
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from PIL import Image, ImageDraw as D
from operator import mul
import json
import os

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
        
        # need to deal with bytes
        elif k == 'image':
            im = Image.fromarray(v)
            global image_count
            os.mkdir('data/' + str(image_count))
            im.save('data/' + str(image_count) + '/im_' + str(image_count) + '.jpeg')
            image_count += 1
            # plt.figure()
            # plt.imshow(image)
            # plt.show()

        else:
            if isinstance(v, np.ndarray):
                v= str(v.tolist())
            elif isinstance(v, bytes):
                v= v.decode("utf-8")
            elif k == 'label':
                #print('begin')
                #print(string_to_array(v))
                #print('end')
                image = example['image']
                # plt.figure()
                # plt.imshow(image)
                # plt.show()
                v = change_label_array(v)
                print(v)
            # dataSet = open("data/" + str(image_count) + "/meta_" + str(image_count) + ".txt", "w")
            # dataSet.write(("{0} : {1}".format(k, v)))
            # dataSet.write("\n")
            # dataSet.close()
    
#%%
ds, info = tfds.load('coco', split='train', with_info=True)
ds = tfds.as_numpy(ds)
file = open(r"/home/ubuntu/tensorflow_datasets/coco/2014/1.1.0/objects-label.labels.txt")
word = file.readlines()
#print(word)
lines = []
image_bytes = open("image_bytes.txt", "wb")
for obj in objects:
    #print(obj)
    for count, value in enumerate(word):
        if obj in value:
            lines.append(count)


#%%
os.mkdir('data')
x = 0
image_count = 0
# remove breakpoint to continue though whole dataset
for example in ds:  # (image[], labels[], objects[], bbox[])
    if x == 15:
        break
    print(example['image'].shape)
    image = example['image']
    labels = example['objects']['label']
    bboxes = example['objects']['bbox'] 

    #print(labels)
    # print(bboxes)  # [y_min, x_min, y_max, x_max]
    if (set(lines) & set(labels)):
        myprint(example)      
    x += 1


file.close()
image_bytes.close()


# %%)