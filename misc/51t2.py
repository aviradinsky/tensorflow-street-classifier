#%%
import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image
import os


# %%
d = foz.list_downloaded_zoo_datasets()
print(d)
# %%
dataset = fo.load_dataset('open-images-v6-train')
# %%
session = fo.launch_app(dataset)
session.dataset = dataset
session.wait()
#%%

z=0
for dir in os.listdir(os.getcwd() + "/data"):
    print(dir)
    for objects in os.listdir(os.getcwd() + "/data/"+ dir):
            print(str(objects) + ":" , len(os.listdir(os.getcwd() + "/data/" + dir + "/" + objects)) ) 
            z+=len(os.listdir(os.getcwd() + "/data/" + dir + "/" + objects))
print(z)
#%%
directory = f'{os.getcwd()}/data'
x =0
for sample in dataset:
    print(x)
    i= Image.open(sample["filepath"])
    height = i.height
    width =i.width
    # i.show()
    #print(sample)
    boxes= sample["detections"]["detections"]
    for box in boxes:
        # print(sample["filepath"])
        label =box['label']
        label =label.lower()
        label=label.replace(" ", "_")
        #print(label)
        # print( box["bounding_box"])
        cropSize = (box["bounding_box"][0]*width,
                    box["bounding_box"][1]*height,
                    box["bounding_box"][2]*width +box["bounding_box"][0]*width ,
                    box["bounding_box"][3]*height +box["bounding_box"][1]*height )
        # print(i.size)
        # print(cropSize)
        crop = i.crop(cropSize)
        if(crop.width<20 or crop.height<20):
            #print("too small")
            continue
        # print(crop.size)
        #crop.show()
        if z % 8 == 0:
            label = f'test/{label}'
        else:
            label = f'train/{label}'
        crop.save(f'{directory}/{label}/{z}.jpg')
        z+=1
    #print("\n")
    x+=1
# %%
w=0
for dir in os.listdir(os.getcwd() + "/data"):
    print(dir)
    for objects in os.listdir(os.getcwd() + "/data/"+ dir):
            print(str(objects) + ":" , len(os.listdir(os.getcwd() + "/data/" + dir + "/" + objects)) ) 
            w+=len(os.listdir(os.getcwd() + "/data/" + dir + "/" + objects))
print(w)