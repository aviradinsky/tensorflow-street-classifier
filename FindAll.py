import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

from PIL import Image, ImageDraw as D
from operator import mul
import json
import pickle

def myprint(d):
    for k, v in d.items():
        if isinstance(v, dict):
            myprint(v)
        
        else:
            if isinstance(v, np.ndarray):
                v= str(v.tolist())
            if isinstance(v, bytes):
                v= v.decode("utf-8")
            dataSet.write(("{0}:{1}".format(k, v)))
            dataSet.write("\n")
    
ds, info = tfds.load('coco', split='train', with_info=True)
ds = tfds.as_numpy(ds)
file = open(r"C:\Users\jakes\tensorflow_datasets\coco\2014\1.1.0\objects-label.labels.txt")
word = file.readlines()
objects = ["bicycle", "motorcycle", "bus", "truck", "car", "train",
           "person", "traffic light", "stop sign", "fire hydrant"]
lines = []
dataSet = open("DataSet.txt", "w")
for obj in objects:
    print(obj)
    for count, value in enumerate(word):
        if obj in value:
            lines.append(count)


x = 0

# remove breakpoint to continue though whole dataset
for example in ds:  # (image[], labels[], objects[], bbox[])
    
    if x == 7:
        break

    image =example['image']
    labels = example['objects']['label']
    bboxes = example['objects']['bbox']
    #print(labels)
    # print(bboxes)  # [y_min, x_min, y_max, x_max]
    if (set(lines) & set(labels)):
        myprint(example)      
    x += 1



