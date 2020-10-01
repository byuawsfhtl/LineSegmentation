import os

import tensorflow as tf
import pandas as pd


def encode_single_img(path, img_size, binary=False, standardization=True):
    """
    This function takes a file path of an image, reads the file, converts to a tensor and performs the necessary
    standardization, resizing, and padding.

    :param path: Path to the image
    :param img_size: The final size of the image after resizing and padding (height, width)
    :param binary: Boolean indicating whether or not the image should be binary (used for ground-truth images)
    :param standardization: Boolean indicating whether or not the image should be standardized using Tensorflow's
                            tf.image.per_image_standardization() function.
    :return: The encoded image
    """
    num_channels = 1 if binary else 3

    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, dtype=tf.float32, channels=num_channels)
    img = tf.image.resize_with_pad(img, img_size[0], img_size[1])

    if standardization:
        img = tf.image.per_image_standardization(img)

    if binary:
        img = tf.where(img > .5, tf.ones_like(img), tf.zeros_like(img))

    return img


def encode_img_with_name(img_path, file_separator, img_size):
    """
    This function takes an image path and returns a tensor image in its desired size with its associated name.

    :param img_path: The path to the image to be loaded, encoded, resized
    :param file_separator: The file separator for the operating system
    :param img_size: The size of the image after resizing
    :return: The encoded image and name
    """
    img = encode_single_img(img_path, img_size, standardization=False)
    img_name = tf.strings.split(tf.strings.split(img_path, sep=file_separator)[-1], sep='.')[0]

    return img, img_name


def encode_imgs(original_path, gt_path, img_size):
    """
    This function takes the original/ground_truth file paths and converts to tensors. See encode_single_img function
    for more details. Note that the ground-truth image is encoded as a binary image.

    :param original_path: The path to the original image to be encoded
    :param gt_path: The path to the ground-truth image to be encoded
    :param img_size: The final size of the images after resizing
    :return: The encoded original and gt images
    """
    original = encode_single_img(original_path, img_size)
    gt = encode_single_img(gt_path, img_size, binary=True)

    return original, gt


def get_dataset_size(csv_path):
    """
    The tf.data api has a hard time producing the dataset size. The cardinality() method often returns unknown even
    with the CsvDataset. This function uses pandas to get the length.

    :param csv_path:  The path to the csv containing information about the dataset
    :return:  The size of the dataset
    """
    return len(pd.read_csv(csv_path, sep='\t', header=None, names=['original_path', 'gt_path']))


def get_encoded_dataset_from_csv(csv_path, img_size):
    """
    Using the tf.data api, load the desired csv with original and ground-truth data, encode the images for use with
    the segmentation model and return the desired tf dataset.

    :param csv_path: The path to the tab-delimited csv file containing | Original Image | Ground Truth |
    :param img_size: The size of the image after resizing/padding (height, width)
    :return: The tf dataset containing encoded images (original, gt)
    """
    path_sep = os.path.sep
    path_prefix = tf.strings.join(csv_path.split('/')[:-1], path_sep)
    return tf.data.experimental.CsvDataset(csv_path, ['original', 'gt'], field_delim='\t', use_quote_delim=False).map(
        lambda orig_path, gt_path: encode_imgs(
            tf.strings.join([path_prefix, tf.strings.reduce_join(tf.strings.split(orig_path, '/'), separator=path_sep)],
                            separator=path_sep),
            tf.strings.join([path_prefix, tf.strings.reduce_join(tf.strings.split(gt_path, '/'), separator=path_sep)],
                            separator=path_sep),
            img_size)
    )


def get_encoded_inference_dataset_from_img_path(img_path, img_size):
    """
    Using the tf.data api, load all the images from the desired path and return a dataset containing encoded images
    and the image name (without path or extension information).

    :param img_path: The path tot he directory containing images
    :param img_size: The size of the image after resizing/padding (height, width)
    :return: The tf dataset containing encoded images and their respective string names
    """
    return tf.data.Dataset.list_files(img_path + '/*', shuffle=False).map(
        lambda path: encode_img_with_name(path, os.path.sep, img_size)
    )
