#%%
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image
import numpy as np
import seaborn as sns
#%%
model = tf.keras.models.load_model('model')
directory = f'{os.getcwd()}/data/'
image_size = (64,64,3)
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=f'{directory}/test',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    image_size=image_size[:2],
    shuffle=True,
    seed=7
)
#%%
test_images = []
test_labels = []
x=0
for image, label in tfds.as_numpy(test_data):
    print(x) #batch
    test_images.append(image)
    test_labels.append(label)
    x+=1

#%%
#del test_images[-1]
#del test_labels[-1]
y_pred = []
x=0
for batch in test_images:
    print(x)
    y_pred.append(np.argmax(model.predict(batch), axis=1)) #predict each item in batch
    x+=1
y_true = test_labels
for i in range(len(y_pred)):
    #print(y_pred[i])
    #print(y_true[i])
    test_acc = sum(y_pred[i] == y_true[i]) / len(y_true[i]) 
    print(f'Test set accuracy: {test_acc}') #per batch
#%%
objects = os.listdir(os.getcwd()+"/data/test")
objects.sort()
confusion_mtx= np.zeros((11,11))
for x in range(len(y_pred)): #merge all batches into one matrix
    confusion_mtx += tfds.as_numpy(tf.math.confusion_matrix(y_true[i], y_pred[i])) 
plt.figure(figsize=(11, 11))
sns.heatmap(confusion_mtx, xticklabels= objects, yticklabels= objects, annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()
#%%