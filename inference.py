import sys

import tensorflow as tf
from tqdm import tqdm
import yaml

import lineseg.dataset as ds
from lineseg.model import ARUNet
from lineseg.seg import segment_from_predictions
from lineseg.util import model_inference


IMG_PATH = 'img_path'
OUT_PATH = 'out_path'
MODEL_IN = 'model_in'
SAVE_RAW = 'save_raw'
RAW_PATH = 'raw_path'
IMG_SIZE = 'img_size'
BATCH_SIZE = 'batch_size'
PLOT_IMGS = 'plot_imgs'


def inference(cmd_args):
    """
    Perform inference on images specified by the user

    python inference.py <INFERENCE_CONFIG_FILE>

    Command Line Arguments:
    * INFERENCE_CONFIG_FILE (required): The path to the inference configuration file. An inference configuration
      file is provided as "inference_config.yaml".

    Configuration File Arguments:
    * img_path: The path to the directory of images to be inferred
    * out_path: The path to the directory that segmented line snippets will be stored
    * model_in: The path to the pre-trained model weights
    * save_raw: Whether or not to save the raw output of the semantic segmentation model
    * raw_path: The path to the directory that the raw output images will be stored
    * img_size: The size which all images will be resized for inference
    * batch_size: The size of the mini-batch used during inference
    * seg_step_size: How many pixels along the baseline to look at when searching the image to create a bounding polygon
    * plot_imgs: Whether or not to plot each text line snippet during the segmentation process (used for debugging)

    :param cmd_args: Command line arguments
    :return: None
    """
    # Ensure the inference config file is included
    if len(cmd_args) == 0:
        print('Must include path to inference config file. The default file is included as inference_config.yaml')
        return

    # Read arguments from the config file
    with open(cmd_args[0]) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Create our ARU-Net Models
    model = ARUNet()

    # Load the pre-trained model weights
    model.load_weights(configs[MODEL_IN])

    dataset = ds.get_encoded_inference_dataset_from_img_path(configs[IMG_PATH], eval(configs[IMG_SIZE]))\
        .batch(configs[BATCH_SIZE])
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()

    # Iterate through each of the images and perform inference
    inference_loop = tqdm(total=dataset_size, position=0, leave=True)


    for original_imgs, resized_imgs, img_names in dataset:
        baseline_predictions = model_inference(model, resized_imgs)

        # Iterate over the batch
        for original_img, baseline_prediction, img_name in zip(original_imgs, baseline_predictions, img_names):
            # Segment lines based on the output of the model and save individual line snippets to the given out path
            segment_from_predictions(original_img, baseline_prediction, str(img_name.numpy(), 'utf-8'),
                                     configs[OUT_PATH], plot_images=configs[PLOT_IMGS], include_coords_in_path=True,
                                     save_raw=configs[SAVE_RAW], raw_path=configs[RAW_PATH])
        inference_loop.update(1)

    print('Finished performing inference.')


if __name__ == '__main__':
    inference(sys.argv[1:])
