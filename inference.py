import sys
import os

import tensorflow as tf
from tqdm import tqdm
import yaml

import lineseg.dataset as ds
from lineseg.model import ARUNet
from lineseg.seg import segment_from_predictions
from lineseg.util import model_inference

IMG_PATH = 'img_path'
OUT_PATH = 'out_path'
ORIGINAL_OUT_PATH = 'original_out_path'
MODEL_IN = 'model_in'
SAVE_RAW = 'save_raw'
RAW_PATH = 'raw_path'
IMG_SIZE = 'img_size'
BATCH_SIZE = 'batch_size'
SEG_STEP_SIZE = 'seg_step_size'
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

    original_out_path = configs[ORIGINAL_OUT_PATH] if configs[ORIGINAL_OUT_PATH] else None
    print('Original_out_path:', original_out_path)

    # Load the pre-trained model weights
    model.load_weights(configs[MODEL_IN])

    dataset = ds.get_encoded_inference_dataset_from_img_path(configs[IMG_PATH], eval(configs[IMG_SIZE]))
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()

    # Iterate through each of the images and perform inference
    inference_loop = tqdm(total=dataset_size, position=0, leave=True)
    for img, img_name in dataset:
        std_img = tf.image.per_image_standardization(img)  # The inference dataset doesn't standardize the image input
        baseline_prediction = model_inference(model, tf.expand_dims(std_img, 0))

        # Save the raw model output if specified in configuration file
        if configs[SAVE_RAW]:
            pred = tf.squeeze(tf.argmax(baseline_prediction, 3))  # Get the most likely class (baseline/non-baseline)
            encoded = tf.image.encode_jpeg(tf.expand_dims(tf.cast(pred * 255, tf.uint8), 2))  # Convert to jpeg
            tf.io.write_file(os.path.join(configs[RAW_PATH], str(img_name.numpy(), 'utf-8') + '.jpg'), encoded)

        # Segment lines based on the output of the model and save individual line snippets to the given out path
        segment_from_predictions(img, baseline_prediction, str(img_name.numpy(), 'utf-8'), configs[OUT_PATH],
                                 plot_images=configs[PLOT_IMGS], include_coords_in_path=True,
                                 save_original_image_path=original_out_path)
        inference_loop.update(1)

    print('Finished performing inference.')


if __name__ == '__main__':
    inference(sys.argv[1:])
