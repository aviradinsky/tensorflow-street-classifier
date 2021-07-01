
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json
import os

from PIL import Image


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


image_count = 0



def change_label_array(v):
    v = (v.astype(np.int32))#.tolist()
  
    v =v.tolist()
    for i in range(len(v)):
        v[i] = conversion_dict.get(v[i])
    return v

def saveData(d): # (image: nparray[], image/filename: byte, ,image/id, numpy int64 objects:{area : nparray[]int64, bbox: nparray[], id: nparray[]float32, is_crowd : nparray[] bool, label : nparray[]int64})
    try:
        os.makedirs(os.getcwd()+'/data/' + str(image_count)) #os.getcwd() = current working directory
    except OSError as error:
        print("already dir")
    img= d['image'] 
    raw_lables = change_label_array(d['objects']['label']) # this still has None values within
    raw_bboxes = d['objects']['bbox'].tolist()
    cleaned_bboxes = [box for i, box in enumerate(raw_bboxes) if raw_lables[i] is not None]

    im = Image.fromarray(img)
    im.save('data/' + str(image_count) + '/im_' + str(image_count) + '.jpeg')
    del d['image'] 
    
    d['image/filename'] =  "im_" + str(image_count) + ".json"
    d['image/id']= d['image/id'].item()
    d['objects']['area'] = (d['objects']['area'].astype(np.int32))
   
    d['objects']['area'] = d['objects']['area'].tolist()
   
    d['objects']['bbox'] = cleaned_bboxes
    d['objects']['id']= (d['objects']['id'].astype(np.int32))
   
    d['objects']['id']= d['objects']['id'].tolist()
    d['objects']['is_crowd']= d['objects']['is_crowd'].tolist()
    
    d['objects']['label'] = [i for i in raw_lables if i is not None]
 
    
    dataFile = open(os.getcwd()+"/data/" + str(image_count) + "/data_" + str(image_count) + ".json", "w")
    json.dump(d, dataFile)       
    

ds, info = tfds.load('coco', split='train', with_info=True)
ds = tfds.as_numpy(ds)
path_to_labels = r"/home/ubuntu/tensorflow_datasets/coco/2014/1.1.0/objects-label.labels.txt"
with open(path_to_labels) as labels_file :
    word = labels_file.readlines()
#print(word)
lines = conversion_dict.keys()




# make sure the base directory exists before saving files in it
# ************************************************************
# MAKE SURE TO CHANGE DIRECTORY NAMEs TO THE LOCATION YOU WOULD
# LOKE THE DATASET TO BE STORED
# ************************************************************
x = 0
# remove breakpoint to continue though whole dataset
for example in ds:  # (image[], labels[], objects[], bbox[])
    if x == 7:
        break
    image = example['image']
    labels = example['objects']['label']
    bboxes = example['objects']['bbox'] 

    #print(labels)
    # print(bboxes)  # [y_min, x_min, y_max, x_max]
    if (set(lines) & set(labels)):
        saveData(example)      
        image_count += 1
    x += 1


