from collections import defaultdict
import numpy as np
import json
import os
from IPython.display import display
from PIL import Image

dictionary = defaultdict(int)
dictionaryclasses = defaultdict(int)
conversion_dict = {
    0: 'bike',
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
filenames= conversion_dict.values()
for n in filenames:
    try:    #make file called instances and in it has files for each instance class
        os.makedirs(os.getcwd()+'/instances/' + n) #os.getcwd() = current working directory 
    except OSError as error:
        print("already dir")

x=0
while True:
    print(x)
    d = os.getcwd()+'/data/' + str(x)+ "/data_"+str(x)+ ".json" 
    try:
        file = open( d)
    except OSError as error:
        print(d)
        break
    image = Image.open(os.getcwd()+'/data/' + str(x)+ "/im_"+str(x)+ ".jpeg" )
    #image.show()
    height = image.height
    width = image.width
   
    data = json.load(file)
    labels= data['objects']['label']
    boxes = data['objects']['bbox'] # [y_min, x_min, y_max, x_max]
    bbox = []
    broken =[]
    for b in boxes:
        bbox.append([int(b[1]*width), int(b[0]*height),int(b[3]*width), int(b[2]*height)] ) #(x, y, x, y)
    boxandlabels = zip(labels,bbox)
    i=0
    
    for l, b in boxandlabels: 
        if(b[0]==b[2] or b[1]==b[3]):
            broken.append(b)   
            print(b)
            continue
        location = str(os.getcwd()+'/instances/' + conversion_dict[l])
        imageName = "/im_"+str(x)+ "_"+str(i)+".jpeg"
       
        crop = image.crop(b) #crops image
        #crop.show()
        
        crop.save(location+imageName) #saves image
        i+=1
    x+=1

print(broken)       