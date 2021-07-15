# %%
import os
from PIL import Image


# %%
train = os.getcwd()+"/data/train/background"
test = os.getcwd()+"/data/test/background"
#%%
for imagename in os.listdir(train):
    print(imagename)
    image = Image.open(train+"/"+ imagename)
    height, width = image.size
    i1 = image.crop((0,0, height/2, width/2))
    i2 = image.crop((height/2,0, height, width/2))
    i3 = image.crop((0,width/2, height/2, width))
    i4 = image.crop((height/2, width/2, height,width))
    imagename= imagename.replace(".jpg","")   
    i1.save(train+"/"+ imagename+"_"+str(1)+".jpg")
    i2.save(train+"/"+ imagename+"_"+str(2)+".jpg")
    i3.save(train+"/"+ imagename+"_"+str(3)+".jpg")
    i4.save(train+"/"+ imagename+"_"+str(4)+".jpg")
    os.remove(train+"/"+ imagename+".jpg")
    
# %%
files =os.listdir(test)
files.sort()
print(files)
for imagename in files:
    print(imagename)
    image = Image.open(test+"/"+ imagename)
    height, width = image.size
    i1 = image.crop((0,0, height/2, width/2))
    i2 = image.crop((height/2,0, height, width/2))
    i3 = image.crop((0,width/2, height/2, width))
    i4 = image.crop((height/2, width/2, height,width))
    imagename= imagename.replace(".jpg","")   
    i1.save(test+"/"+ imagename+"_"+str(1)+".jpg")
    i2.save(test+"/"+ imagename+"_"+str(2)+".jpg")
    i3.save(test+"/"+ imagename+"_"+str(3)+".jpg")
    i4.save(test+"/"+ imagename+"_"+str(4)+".jpg")
    os.remove(test+"/"+ imagename+".jpg")

# %%
