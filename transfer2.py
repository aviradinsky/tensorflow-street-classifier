# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.keras.engine import training
from params import image_size, model_dir, new_labels
# %%
class_number = len(new_labels) # for background
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
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
# %%
train_generator = train_datagen.flow_from_directory(
    './newdata/train', 
    target_size = image_size,
    color_mode = 'rgb',
    batch_size = 64,
    class_mode = 'categorical',
    shuffle = True
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
history = model.fit_generator(generator = train_generator, steps_per_epoch=train_generator.n//train_generator.batch_size, epochs = epochs)
# %%
model.save(model_dir)
# %%
import confusionMatrix
# %%