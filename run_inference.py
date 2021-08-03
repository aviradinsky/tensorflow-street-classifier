# %%

from inference import infer_ss
from PIL import Image
import tensorflow_datasets as tfds
from inference import display_image as di
import os
import params

# %%

# %%

labels = {
    1 : 'person',
    2 : 'bicycle',
    3 : 'car',
    7 : 'train'
}

# load test data
data = tfds.load('coco').get('train')

# collect test images
images=[]
count = 0
for sample in data:
    if count >= 30:
        break
    objects = sample.get('objects')
    all_labels = set(objects.get('label').numpy())
    # add image if it contains labels that we are using
    if not all_labels.isdisjoint(labels):
        count += 1
        image = sample.get('image').numpy()
        images.append(image)
    

# %%
# run inference on each image
model_dir = params.model_dir
results = []

for img in images:
    results.append(infer_ss(model_dir, img, display_img=False))

#  %%
# save plain image and image with bboxes to directory
directory = './inference_test'

if not os.path.exists(directory):
    os.makedirs(directory)

for i, result in enumerate(results):
    if not os.path.exists(f'{directory}/{i}'):
        os.makedirs(f'{directory}/{i}')
    bbox_img = Image.fromarray(result)
    bbox_img.save(f'{directory}/{i}/bbox.jpg')
    plain_img = Image.fromarray(images[i])
    plain_img.save(f'{directory}/{i}/plain.jpg')
    

# %%
