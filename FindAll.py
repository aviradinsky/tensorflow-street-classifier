#%%
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

from PIL import Image, ImageDraw as D
from operator import mul
import json
import pickle
import matplotlib.pyplot as plt

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
        
        else:
            if isinstance(v, np.ndarray):
                v= str(v.tolist())
            if isinstance(v, bytes):
                v= v.decode("utf-8")
            if k == 'label':
                #print('begin')
                #print(string_to_array(v))
                #print('end')
                image = example['image']
                plt.figure()
                plt.imshow(image)
                plt.show()
                v = change_label_array(v)
                print(v)

            dataSet.write(("{0} : {1}".format(k, v)))
            dataSet.write("\n")
    
#%%
ds, info = tfds.load('coco', split='train', with_info=True)
ds = tfds.as_numpy(ds)
file = open(r"/home/ubuntu/tensorflow_datasets/coco/2014/1.1.0/objects-label.labels.txt")
word = file.readlines()
#print(word)
lines = []
dataSet = open("DataSet.txt", "w")
for obj in objects:
    #print(obj)
    for count, value in enumerate(word):
        if obj in value:
            lines.append(count)


#%%
x = 0
# remove breakpoint to continue though whole dataset
for example in ds:  # (image[], labels[], objects[], bbox[])
    if x == 20:
        break

    image = example['image']
    labels = example['objects']['label']
    bboxes = example['objects']['bbox'] 

    #print(labels)
    # print(bboxes)  # [y_min, x_min, y_max, x_max]
    if (set(lines) & set(labels)):
        myprint(example)      
    x += 1


file.close
dataSet.close
