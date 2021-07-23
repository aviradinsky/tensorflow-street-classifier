# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.keras.engine import training
from params import image_size, label_list, model_dir, new_labels_list
import load_data
import confusionMatrix as cm
# %%
load_data.main()
# %%
class_number = len(new_labels_list) # for background

IMG_SIZE = image_size[:2]
BATCH_SIZE = 32
# %%
model = tf.keras.applications.ResNet50(weights='imagenet')
base_model = tf.keras.applications.ResNet50(weights='imagenet',include_top=False)
preprocess_input = tf.keras.applications.resnet50.preprocess_input
# %%
inputs = tf.keras.Input(shape=image_size)
x = preprocess_input(inputs)
x = base_model(x, training, False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
preds = tf.keras.layers.Dense(class_number, activation ='softmax')(x)
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
    batch_size = 64,
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
"""
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_generator = train_generator.prefetch(buffer_size = AUTOTUNE)
"""
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
cm.matrix('newModelTransfer')

#%%