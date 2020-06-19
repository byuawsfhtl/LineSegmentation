import sys

import tensorflow as tf
from tqdm import tqdm

from src.lineseg.model import ARUNet
from src.lineseg.dataset.sequence import ARUSequence
from src.lineseg.util.arguments import IArg, InfArgParser
from src.lineseg.seg import segment_from_predictions


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

    args = InfArgParser(cmd_args)
    args.parse()

    sequence = ARUSequence(args[IArg.IMG_PATH])
    baseline_model = ARUNet()
    seam_model = ARUNet()

    # Load the pre-trained model weights
    baseline_model.load_weights(args[IArg.WEIGHTS_PATH_BASELINE])
    seam_model.load_weights(args[IArg.WEIGHTS_PATH_SEAM])

    # Iterate through each of the images and perform inference
    for img, img_name in tqdm(sequence):
        baseline_prediction = baseline_model(tf.expand_dims(img, 0))
        seam_prediction = seam_model(tf.expand_dims(img, 0))

        segment_from_predictions(img, baseline_prediction, seam_prediction, img_name,
                                 int(args[IArg.SEGMENTATION_STEP_SIZE]),
                                 plot_images=args[IArg.SHOULD_PLOT_IMAGES] == 'True', save_path=args[IArg.OUT_PATH])

    print('Finished performing inference.')


if __name__ == '__main__':
    inference(sys.argv[1:])
