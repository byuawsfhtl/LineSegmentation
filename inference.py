import sys
import os

import tensorflow as tf
from tqdm import tqdm
import yaml

import lineseg.dataset as ds
from lineseg.model import ARUNet
from lineseg.seg import segment_from_predictions

IMG_PATH = 'img_path'
OUT_PATH = 'out_path'
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
    * img_path: The path to the images to be inferred
    * out_path: The path to the results of the inference (text-line snippets)
    * model_in: The path to the pre-trained model weights
    * img_size: The height and width for resizing the image when sent into the model for inference
                (default: 768, 1152)
    * seg_step_size: How many columns along the baseline to look at when searching the seam image to find
                     the bounding polygon (default: 1)
    * plot_imgs: Should each text line snippet be shown to the screen during inference? (default: False)

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

    dataset = ds.get_encoded_inference_dataset_from_img_path(configs[IMG_PATH], eval(configs[IMG_SIZE]))
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()

    # Iterate through each of the images and perform inference
    inference_loop = tqdm(total=dataset_size, position=0, leave=True)
    for img, img_name in dataset:
        std_img = tf.image.per_image_standardization(img)  # The inference dataset doesn't standardize the image input
        baseline_prediction = model(tf.expand_dims(std_img, 0), training=True)

        # Save the raw model output if specified in configuration file
        if configs[SAVE_RAW]:
            pred = tf.squeeze(tf.argmax(baseline_prediction, 3))
            encoded = tf.image.encode_jpeg(tf.expand_dims(tf.cast(pred, tf.uint8), 2))
            tf.io.write_file(os.path.join(configs[RAW_PATH], str(img_name.numpy(), 'utf-8') + '.jpg'), encoded)

        # Segment lines based on the output of the model and save individual line snippets to the given out path
        segment_from_predictions(img, baseline_prediction, str(img_name.numpy(), 'utf-8'), configs[OUT_PATH],
                                 plot_images=configs[PLOT_IMGS])
        inference_loop.update(1)

    print('Finished performing inference.')


if __name__ == '__main__':
    inference(sys.argv[1:])
