# %%
import os
object = {
    0:   'bicycle',
    1:   'motorcycle',
    2:   'bus',
    3:   'truck',
    4:   'car',
    5:   'train',
    6:   'person',
    7:   'traffic_light',
    8:   'stop_sign',
    9:   'fire_hydrant',
    10: 'background'
    }

for i in range(11):
    os.rename(os.getcwd() + "/data/test/"+ str(i), os.getcwd() + "/data/test/"+ object[i] )
    os.rename(os.getcwd() + "/data/train/"+ str(i), os.getcwd() + "/data/train/"+ object[i] )
# %%
