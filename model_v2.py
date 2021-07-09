# %%
import tensorflow as tf
import tensorflow_datasets
import os
# %%
d = tensorflow_datasets.load('coco').get('test')
thing = None
t = d.take(100)
for i in t:
    print(i.get('objects').get('label'))
#%%

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

    if target_height == 0 or target_width == 0 or top_line == 0 or right_line == 0:
        print('garbage is going in')

    cropped_image = tf.image.crop_to_bounding_box(image,
                                                    offset_height=offset_height,
                                                    offset_width=offset_width,
                                                    target_height=target_height,
                                                    target_width=target_width
                                                    )
    
    return cropped_image

def load_data(key: str) -> tf.data.Dataset:

    directory = f'{os.getcwd()}/data/{key}'

    if os.path.exists(f'{directory}/0'):
        # this means that the data was already loaded
        print(f'Data of type {key} is already in the data folder')
    else:
        for i in range(len(chosen_labels) + 1):
            os.makedirs(f'{directory}/{i}')
        count = 0
        dataset = tensorflow_datasets.load('coco').get(key)        
        for data in dataset:
            image = data.get('image')
            objects = data.get('objects')
            all_labels = objects.get('label').numpy()
            if len(set(all_labels) & set(chosen_labels)) == 0:

                tf.keras.preprocessing.image.save_img(f'{directory}/{len(chosen_labels)}/{count}.png',image)
                count += 1
                continue
            else:
                for ind, label in enumerate(all_labels):
                    if label in chosen_labels:
                        image = crop_tensor_by_nth_bbox(data,ind)
                        tf.keras.preprocessing.image.save_img(f'{directory}/{chosen_labels.index(label)}/{count}.png',image)
                        count += 1


    
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory=directory,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        seed=6
    )


valid_keys = ('train','validation','test2015')

test_data = load_data(valid_keys[2])


# %%
