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
file = open("small.txt","w")
for name in conversion_dict.values():
    print(name)
    directory = os.getcwd()+'/instances/' + name
    for file in os.listdir(directory):
        try:
            image = Image.open(directory+"/" +file)
            if(image.height<5 or image.width<5):
                print(directory+"/" +file,image.height,image.width )
                
        except:
            print(directory+"/" +file)
            os.remove(directory+"/" +file)
 
