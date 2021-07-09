# %%
import numpy as np
import json
import os

from PIL import Image
from numpy import random

for i in range(11):
    print(i)
    try:
        os.makedirs('cropped_images/train/'+i)
        os.makedirs('cropped_images/val/'+i)
        os.makedirs('cropped_images/test/'+i)
    except IOError as e:
        print("already dir")
    
    arr= os.listdir('cropped_images/'+i)
    arr.sort()  # make sure that the filenames have a fixed order before shuffling
    random.seed(230)
    random.shuffle(arr)
    train = int(.8*len(arr))
    val = int(.9*len(arr))
    test = len(arr)
    train_data= arr[0:train]
    val_data= arr[train:val]
    test_data= arr[val:test]
    for trainName in train_data:
        print(trainName)
        os.rename('cropped_images/'+i+'/'+ trainName,'cropped_images/train/'+i+'/'+ trainName)
    for valName in val_data:
        print(valName)
        os.rename('cropped_images/'+i+'/'+ valName,'cropped_images/val/'+i+'/'+ valName)  
    for testName in test_data:
        print(testName)
        os.rename('cropped_images/'+i+'/'+ testName,'cropped_images/test/'+i+'/'+ testName)  
    os.rmdir('cropped_images/'+i) 

# %%
for i in  range(11):
    tr= os.listdir('cropped_images/train/'+str(i))
    va= os.listdir('cropped_images/val/'+str(i))
    te = os.listdir('cropped_images/test/'+str(i))
    lenght =len(tr)+ len(va)+ len(te)
    print(len(tr), len(va), len(te))
    print(len(tr)/lenght, len(va)/lenght, len(te)/lenght)
# %%
