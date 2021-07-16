# %%
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten
from numpy import random
import numpy as np
from PIL import Image, ImageDraw
import scale_and_slide
import params


# %%
model = tf.keras.models.load_model('model')
model.summary()

# %%

x=0
images=[]
for data in tfds.as_numpy(tfds.load('coco').get('test2015')):
    #print(data)
    if(x>5):
        break

    image = Image.fromarray(data['image']) 
    #image.show()
    #print(data['image/filename'].decode("utf-8") )
    images.append(image)
    x+=1
# %%

window_dimensions = (75, 100)
stride = 50
num_rescales = 3
rescale_increment = .5
img =images[1]
#crops = scale_and_slide.get_image_chunks(img, window_dimensions, stride, num_rescales, rescale_increment)
crops_set = scale_and_slide.sliding_window(img, window_dimensions, stride)
#rescaled_images = scale_and_slide.get_scaled_images(img, num_rescales, rescale_increment)
scale_and_slide.simple_display_image(img)
# %%
data = params.chosen_labels
data.append("background")
data.sort()
print(data)
for i in crops_set:
    #print(i[0])
    #scale_and_slide.display_crops(rescaled_images[i], crops[i], window_dimensions, stride)
    i= i[0].resize((100,100))
    plt.figure()
    
    
    
    obj= tf.keras.preprocessing.image.img_to_array(i)
    obj = tf.expand_dims(obj, 0) # Create a batch
    predictions = model.predict(obj)
    # print(predictions.shape)
    # print(predictions)
   
    score = tf.nn.softmax(predictions[0])
    #print(score)
    
    #print(obj)
    
    plt.imshow(i)
    plt.title((" {} with {:.2f} percent confidence.".format(data[np.argmax(score)], 100 * np.max(score))))
    plt.show()
    
# %%
