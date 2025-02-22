#%%
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from params import new_labels_list, data_dir, model_dir
#%%
def matrix(modelname):
    model = tf.keras.models.load_model(modelname)
    directory = f'{os.getcwd()}/{data_dir}/'
    image_size = (160,160,3)
    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        directory=f'{directory}/test',
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        image_size=image_size[:2],
        shuffle=True,
        seed=7
    )

    test_images = []
    test_labels = []
    x=0
    for image, label in tfds.as_numpy(test_data):
    #print(x) #batch
        test_images.append(image)
        test_labels.append(label)
        x+=1

  

    y_pred = []
    x=0
    for batch in test_images:
    #print(x)
        y_pred.append(np.argmax(model.predict(batch), axis=1)) #predict each item in batch
        x+=1
    y_true = test_labels
    for i in range(len(y_pred)):
        #print(y_pred[i])
        #print(y_true[i])
        test_acc = sum(y_pred[i] == y_true[i]) / len(y_true[i]) 
        print(f'Test set accuracy: {test_acc}') #per batch

    objects = new_labels_list
    confusion_mtx= np.zeros((6,6))
    for x in range(len(y_pred)): #merge all batches into one matrix
        confusion_mtx += tfds.as_numpy(tf.math.confusion_matrix(y_true[x], y_pred[x], num_classes=6)) 
    fig = plt.figure(figsize=(10, 10))
    confusion_mtx = confusion_mtx/confusion_mtx.sum(axis=1)[:, tf.newaxis]
    sns.heatmap(confusion_mtx, xticklabels= objects, yticklabels= objects, annot=True, fmt='.2%')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()
    fig.savefig('confusion_matrix.jpg')
  
#%%

if __name__ == '__main__':
    matrix(model_dir)

# %%

