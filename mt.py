#%%
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory
import params

#%%
model = tf.keras.models.load_model('newModelTransfer')
directory = f'{os.getcwd()}/newdata/'
image_size = (100,100,3)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
# %%
directory = f'{os.getcwd()}/newdata'
test_data= train_datagen.flow_from_directory(
    f'{directory}/test', 
    target_size = (100, 100),
    color_mode = 'rgb',
    batch_size = 32,
    class_mode = 'categorical',
    shuffle = True,
    
)
# %%
test_loss, test_acc = model.evaluate(test_data, verbose=2)
print(f'test_acc = {test_acc}')
# %%
