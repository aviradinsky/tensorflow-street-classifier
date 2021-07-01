
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from PIL import Image, ImageDraw as D
from operator import mul
import json
import os


objects = ["bicycle", "motorcycle", "bus", "truck", "car", "train",
           "person", "traffic light", "stop sign", "fire hydrant"]
conversion_dict = {
    1: 0,  # bike
    3: 1,  # motorcycle
    5: 2,  # bus
    7: 3,  # truck
    2: 4,  # car
    6: 5,  # train
    0: 6,  # person
    9: 7,  # traffic light
    11: 8,  # stop sign
    10: 9  # fire hydrant
}


ds, info = tfds.load('coco', split='train', with_info=True)
ds = tfds.as_numpy(ds)
file = open(
    r"C:\Users\jakes\tensorflow_datasets\coco\2014\1.1.0\objects-label.labels.txt")
word = file.readlines()
# print(word)
lines = []
jsonFile = open("writeData.json", "w")
for obj in objects:
    # print(obj)
    for count, value in enumerate(word):
        if obj in value:
            lines.append(count)


# os.mkdir('data')
x = 0

# remove breakpoint to continue though whole dataset
dictionarylist= []
for example in ds:  # (image[], labels[], objects[], bbox[])
    if x == 7:
        break
    print(example['image'].shape)
    image = example['image']
    labels = example['objects']['label']
    bboxes = example['objects']['bbox']

    # print(labels)
    # print(bboxes)  # [y_min, x_min, y_max, x_max]
    
    if (set(lines) & set(labels)):
        newexample={}
        # *****************************************
        # save image somewhere else !!!!!
        # ***********************************************
        del example['image'] 
        for k, v in example.items():
            
            if isinstance(v, np.ndarray):
                print(v.dtype)
                if v.dtype is np.dtype(np.int64):
                    print("yay")
                    newVal = val.astype(np.int32)
                    newexample.update({k : newVal.tolist()})
                else:
                    newexample.update({k : v.tolist()})    
            elif isinstance(v, bytes):
                newexample.update({k: v.decode("utf-8")}) 
              
            elif isinstance(v, dict):
                dictionary ={}
                for key, val in v.items():
                    
                    if isinstance(val, np.ndarray):
                        print(val.dtype)
                        if val.dtype is np.dtype(np.int64):
                            
                            newVal = val.astype(np.int32)
                            print(newVal.dtype)
                            dictionary.update({key : newVal.tolist()})
                        else:
                            print("old")
                            dictionary.update({key : val.tolist()})
                    elif isinstance(val, bytes):
                        dictionary.update({key :val.decode("utf-8")}) 
                newexample.update({k : dictionary})
        
        dictionarylist.append(newexample)           
 
    x += 1

json.dump(dictionarylist, jsonFile)

file.close()
jsonFile.close()



