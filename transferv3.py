#%%
import matplotlib.pyplot as plt
from params import image_size, model_dir
import tensorflow as tf
import os
#%%
print(image_size[:2])
#%%
trainval_dir = os.path.join('newdata', 'train')
num_classes = len(os.listdir(trainval_dir))
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory = trainval_dir,
    labels = 'inferred',
    image_size = image_size[:2],
    shuffle = True,
    subset = 'training',
    validation_split = 0.2,
    seed = 6
)
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory = trainval_dir,
    labels = 'inferred',
    image_size = image_size[:2],
    subset = 'validation',
    shuffle = True,
    validation_split = 0.2,
    seed = 6
)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
base_model = tf.keras.applications.MobileNetV2(input_shape=image_size,
                                               include_top = False,
                                               weights = 'imagenet')
base_model.trainable = False
inputs = tf.keras.Input(shape=image_size)
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes)(x)
model = tf.keras.Model(inputs, outputs)
base_learning_rate = 0.0001
model.compile(optimizer = tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
initial_epochs = 5
history = model.fit(train_dataset, epochs = initial_epochs,
                    validation_data = val_dataset)
"""
## Now do fine-tuning:
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])
fine_tune_epochs = 0
total_epochs = initial_epochs + fine_tune_epochs
history_fine = model.fit(train_dataset, epochs=total_epochs, initial_epoch=history.epoch[-1],
                         validation_data=val_dataset)
"""
# %%
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory='newdata/test',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    image_size=image_size[:2],
    shuffle=True
)
test_loss, test_acc = model.evaluate(test_data, verbose=2)
print(f'test_acc = {test_acc}')
#%%
model.save(model_dir)
import confusionMatrix
# %%