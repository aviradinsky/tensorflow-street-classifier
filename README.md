# About

Tensorflow Street Classifier is a repo containing the ML project of YU Students [Nissim Cantor](https://www.linkedin.com/in/nissim-cantor-b1988b19a/), 
[Avi Radinsky](https://www.linkedin.com/in/avi-radinsky/), and [Jacob Silbiger](https://www.linkedin.com/in/jacob-silbiger-0521081b5/), with the help 
from our mentor, [Gershom Kutliroff](https://www.linkedin.com/in/gershom-kutliroff-9a89611/). Our objective was to create a model similar to the ones 
used in self-driving cars, so we decide that our goal should be to identify five classes of objects commonly found on the street: bicycle, car, motorcycle,
person, and train. Using the COCO and Open Images datasets, we developed and trained a transfer learning model in Tensorflow with test accuracy over 95%.

# Project presentation

Checkout our presentation video on youtube: [https://www.youtube.com/watch?v=ItXdPJ3okMo&t=2s](https://www.youtube.com/watch?v=ItXdPJ3okMo&t=2s)\
Download our slide-deck from the presentation: [Computer Vision TF Presentation.pdf](https://github.com/ndcantor/tensorflow-street-classifier/files/6940793/Computer.Vision.TF.Presentation.pdf)

# Build instructions

## Pre-requisites
Make sure to have the following installed and running on your computer:

- Python 3.7
- [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive)
- [Cudnn 7.5.1](https://developer.nvidia.com/cudnn)

## Build the model
Start by cloning the tensorflow-street-classifier repo to your local machine and cd into the repo directory:\
\
&nbsp;&nbsp;&nbsp;&nbsp; `git clone https://github.com/ndcantor/tensorflow-street-classifier.git`\
&nbsp;&nbsp;&nbsp;&nbsp; `cd tensorflow-street-classifier`\
\
Install the required python libaries with pip using our requirements file:\
\
&nbsp;&nbsp;&nbsp;&nbsp; `pip install -r requirements.txt`\
\
To download the datasets, build and train the model, run inference, and get outputs of sample test images (stored in the inferece_test directory and in the confusion_matrix.jpg 
file), run the `street_classifier.py` script (Warning: this can take several hours or more to run):\
\
&nbsp;&nbsp;&nbsp;&nbsp; `python3 street_classifier.py`\
\
Alternatively, you can build and run the model step by step.\
Start by loading the data (if the data has already been loaded, the execution of this script will terminate):\
\
&nbsp;&nbsp;&nbsp;&nbsp; `python3 download_data.py`\
\
Build and train the model:\
\
&nbsp;&nbsp;&nbsp;&nbsp; `python3 transfer_model.py`\
\
Run inference and get sample test outputs:\
\
&nbsp;&nbsp;&nbsp;&nbsp; `python3 run_inference.py`

# Using our pre-trained model
We provided an already trained model for you to use if you don't want to spend the time (and space!) downloading the dataset and training the model.
To run inference on this model or to generate a confusion matrix to see its performance, open the params.py file and changed the assigned value
of the `model_dir` variable to the string `'trained_model'`. After that, you can run the `confusion_matrix.py` and `run_inference.py` scripts from the
terminal directly, and you will then get the results from the pre-trained model.

# Other scripts
To see how our scaling windows or selective search works, run the `scale_and_slide.py` and `selective_search.py` in Visual Studio Code using the Jupyter Notebook extension, and
sample outputs will be displayed. To count the number of images that the model is using to train and test, run `count_data.py`. We also included our old model (based on the [VGG Net](https://en.everybodywiki.com/VGG_Net) architecture) if you wanted to compare the results with our newer transfer learning model. To run that model, run the
`old_vgg_model.py` script from the terminal.

# Thank you
Once again, we would like to thank our mentor, Gershom Kutliroff, for taking the time to help us with this project. We would also like to thank you for checking out our work! 
Please contact us if you have any questions or suggestions on how we can improve our model.
