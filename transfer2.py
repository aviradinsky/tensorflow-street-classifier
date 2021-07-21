# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory
from params import image_size, label_list, model_dir, new_labels_list
import load_data
# %%
load_data.main()
# %%
class_number = len(new_labels_list) # for background
IMG_SIZE = image_size[:2]
BATCH_SIZE = 32
# %%
model = tf.keras.applications.ResNet50(weights='imagenet')
# %%
base_model = tf.keras.applications.ResNet50(weights='imagenet',include_top=False)
# %%
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# %%
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
preds = tf.keras.layers.Dense(class_number, activation ='softmax')(x)
# %%
model = tf.keras.models.Model(inputs=base_model.input, outputs=preds)
# %%
"""
making the first 140 layers untrainable and the last ones trainable
"""
braek = 140
for layer in model.layers[:braek]:
    layer.trainable = False
for layer in model.layers[braek:]:
    layer.trainable = True
# %%
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input,validation_split=0.2)
# %%
directory = f'{os.getcwd()}/newdata'
train_generatorTrain = train_datagen.flow_from_directory(
    f'{directory}/train', 
    target_size = (100, 100),
    color_mode = 'rgb',
    batch_size = 32,
    class_mode = 'categorical',
    shuffle = True,
    subset='training',
    
)
train_generatorVal = train_datagen.flow_from_directory(
    f'{directory}/train', 
    target_size = (100, 100),
    color_mode = 'rgb',
    batch_size = 32,
    class_mode = 'categorical',
    shuffle = True,
    subset='validation',
    
)
# %%
print(model.summary())
# %%
model.compile(
    optimizer='Adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)
# %%
epochs = 15
history = model.fit(train_generatorTrain, 
                    steps_per_epoch=train_generatorTrain.n//train_generatorTrain.batch_size, 
                    validation_data=train_generatorVal,
                    epochs = epochs)
# %%
model.save('newModelTransfer')
#%%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
# %%
import confusionMatrix

#%%