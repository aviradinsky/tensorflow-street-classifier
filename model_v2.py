# %%
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
# %%
"""
global variables
"""
directory = f'{os.getcwd()}/data/'
chosen_labels = [
    1,  # bike
    3,  # motorcycle
    5,  # bus
    7,  # truck
    2,  # car
    6,  # train
    0,  # person
    9,  # traffic light
    11,  # stop sign
    10  # fire hydrant
]
# %%

def crop_tensor_by_nth_bbox(tensor, nth_bbox):
    """
    given an image with bboxs this method returns the cropped image
    for the nth bbox
    """


    bbox = tensor.get('objects').get('bbox').numpy()[nth_bbox]
    image = tensor.get('image')

    top_line = bbox[0]*image.shape[0]
    left_line = bbox[1]*image.shape[1]
    bottom_line = bbox[2]*image.shape[0]
    right_line = bbox[3]*image.shape[1]

    offset_height = int(top_line)
    offset_width = int(left_line)
    target_height = int(bottom_line - top_line)
    target_width = int(right_line - left_line)

    """
    if target_height == 0 or target_width == 0 or top_line == 0 or right_line == 0:
        print('garbage is going in')
    """
    if offset_height == 0 or offset_width == 0 or target_height == 0 or target_width == 0:
        return None
    else:
        return tf.image.crop_to_bounding_box(image,
                                                    offset_height=offset_height,
                                                    offset_width=offset_width,
                                                    target_height=target_height,
                                                    target_width=target_width
                                                    )

def slice_into_4ths(image) -> list:
    x,y,z = image.shape

    x,y = int(x/2),int(y/2)

    top_left = tf.slice(image, [0, 0, 0], [x, y, 3])
    bottom_left = tf.slice(image, [x, 0, 0], [x, y, 3])
    top_right = tf.slice(image, [0, y, 0], [x, y, 3])
    bottom_right = tf.slice(image, [x, y, 0], [x, y, 3])

    return [
        top_right,
        top_left,
        bottom_left,
        bottom_right
    ]



def get_image_manipulations_and_path(image,count) -> list:
    tensor = image
    objects = image.get('objects')
    all_labels = objects.get('label').numpy()
    image = image.get('image') 
    ret = []
    path_buffer: str= ''
    if len(set(all_labels) & set(chosen_labels)) == 0:
        """
        this is background and will be randomly cropped to increase number
        """
        images = slice_into_4ths(image)
        for i in images:
            if count % 8 == 0:
                path = f'{directory}/test/10/{count}.jpeg'
            else:
                path = f'{directory}/train/10/{count}.jpeg'
            count += 1
            ret.append((i,path))
    else:
        """
        this is images that have objects we want, they will be cropped
        """
        for ind, label in enumerate(all_labels):
            if label in chosen_labels:
                image = crop_tensor_by_nth_bbox(tensor,ind)
                if image is None:
                    continue
                if count % 8 == 0:
                    path = f'{directory}/test/{chosen_labels.index(label)}/{count}.jpeg'
                else:
                    path = f'{directory}/train/{chosen_labels.index(label)}/{count}.jpeg'
                count += 1
                ret.append((image,path))
    return (ret,count)


def set_data_in_directories() -> None:
    """
    checking to see if the path ('data/' and 'data/test') exists, if it does no code will run
    """

    if os.path.exists(f'{directory}/train/0'):
        # this means that the data was already loaded
        print(f'Data is already in the data folders')
        return
    else:
        for i in range(len(chosen_labels) + 1):
            os.makedirs(f'{directory}/train/{i}')
            os.makedirs(f'{directory}/test/{i}')
    
    """ 
    there are 4 keys but test and test2015 dont have labels
    """


    valid_keys = ('train','validation')
    count = 0
    track = 0
    for key in valid_keys:
        for image in tfds.load('coco',shuffle_files=True).get(key):
            data = get_image_manipulations_and_path(image,count)
            count = data[1] 
            for manipulation, path in data[0]:
                tf.keras.preprocessing.image.save_img(path, manipulation)

            """
            after this line is code for houskeeping things
            """
            if count > track:
                print(f'{count = }')
                track += 1000
            """
            if count > 100000:
                print('done loading data into dirs')
                return
            """
    pass
# %%
set_data_in_directories()
# %%
image_size = (64,64,3)
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=f'{directory}/train',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    image_size=image_size[0:2],
    shuffle=True,
    subset='training',
    validation_split=0.2,
    seed=6
)
validate_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=f'{directory}/train',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    image_size=image_size[0:2],
    shuffle=True,
    subset='validation',
    validation_split=0.2,
    seed=6
)
# %%
num_classes = len(chosen_labels) + 1 # for the background
data_augmentation = Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal')
])
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=image_size),
    data_augmentation,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes)
])
# %%
model.summary()
# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])
# %%
epochs = 15
history = model.fit(
    train_data,
    validation_data=validate_data,
    epochs=epochs
)
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
#%%
model.save('model')
#%%
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=f'{directory}/test',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    image_size=image_size[:2],
    shuffle=True,
    seed=7
)
test_loss, test_acc = model.evaluate(test_data, verbose=2)
print(f'{test_acc = }')
# %%