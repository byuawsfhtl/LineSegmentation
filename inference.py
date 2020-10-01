import sys

import tensorflow as tf
from tqdm import tqdm
import yaml

import lineseg.dataset as ds
from lineseg.model import ARUNet
from lineseg.seg import segment_from_predictions_without_seam

IMG_PATH = 'img_path'
OUT_PATH = 'out_path'
MODEL_IN = 'model_in'
IMG_SIZE = 'img_size'
BATCH_SIZE = 'batch_size'
SEG_STEP_SIZE = 'seg_step_size'
PLOT_IMGS = 'plot_imgs'


def inference(cmd_args):
    """
    Perform inference on images specified by the user

    python inference.py <ARGS> ...

    Command Line Arguments:
    * img_path (required): The path to the images to be inferred
    * out_path (required): The path to the results of the inference (text-line snippets)
    * weights_path_baseline (required): The path to the pre-trained model weights for baselines
    * weights_path_seam (required): The path to the pre-trained model weights for seams
    * seg_step_size (optional): How many columns along the baseline to look at when searching the seam image to find
                                the bounding polygon (default: 1)
    * plot (optional): Should each text line snippet be shown to the screen during inference? (default: False)
    * image_resize (optional): The height and width for resizing the image when sent into the model for inference
                               (default: 768, 1152)

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
    for img, img_name in dataset:
        print('IMG SHAPE:', img.shape)
        std_img = tf.image.per_image_standardization(img)  # The inference dataset doesn't standardize the image input
        baseline_prediction = model(std_img, training=True)
        segment_from_predictions_without_seam(img, baseline_prediction, img_name, plot_images=configs[PLOT_IMGS],
                                              save_path=configs[OUT_PATH])
        inference_loop.update(1)

    print('Finished performing inference.')


if __name__ == '__main__':
    inference(sys.argv[1:])
