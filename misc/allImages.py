from collections import defaultdict
import numpy as np
import json
import os

from PIL import Image

dictionary = defaultdict(int)
dictionaryclasses = defaultdict(int)
conversion_dict = {
    0:'bike',
    1: 'motorcycle',
    2: 'bus',
    3: 'truck',
    4: 'car',
    5: 'train',
    6: 'person',
    7: 'traffic light',
    8: 'stop sign',
    9: 'fire hydrant',
}
x=0
while True:
    print(x)
    s = os.getcwd()+'/data/' + str(x)+ "/data_"+str(x)+ ".json" 
    try:
        file = open( s)
    except OSError as error:
        print(s)
        break
    data = json.load(file)
    labels= data['objects']['label']
    for value in labels:
        dictionary[conversion_dict[value]] +=1
    for val in set(labels):
        dictionaryclasses[conversion_dict[value]] +=1
    x+=1 
print("there are " + str(x)+ " photos. Labeled 0 - " + str(x-1))
print("number of photos with at least one class: "+ str(dictionaryclasses.items()))            
print("number of instances of a class: "+ str(dictionary.items()))
