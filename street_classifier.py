# %%
# first step - download all the data
import download_data
download_data.main()
# train model, then create and save confusion matrix
import transfer_model
# run inference on model, save sample test images with
# labels to 'inference_test' dir
import run_inference