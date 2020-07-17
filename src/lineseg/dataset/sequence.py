import os

import numpy as np
from PIL import Image
import tensorflow as tf


class LineSequence(tf.keras.utils.Sequence):
    """
    ARU-Sequence

    Keras Sequence class responsible for loading dataset in a TensorFlow compatible format.
    """
    def __init__(self, img_path, label_path=None, mp_layers=10):
        """
        Set up paths and necessary variables.

        :param img_path: Path to the images
        :param label_path:
        """
        self.img_path = img_path
        self.label_path = label_path
        self.divisor = 2 ** mp_layers

        if not os.path.exists(self.img_path):
            raise Exception('Images do not exist in', self.img_path)
        if self.label_path is not None and not os.path.exists(self.label_path):
            raise Exception('Labels do not exist in', self.label_path)

        self.imgs = os.listdir(self.img_path)

    def closest_multiple(self, value):
        q = value // self.divisor
        n1 = q * self.divisor
        n2 = (q + 1) * self.divisor

        if np.abs(n1 - value) < np.abs(n2 - value):
            return n1
        else:
            return n2

    def tensor_image(self, path, pil_format):
        """
        Load an image from the given path, resize it, and convert it to a tensor. The PIL format is helpful
        to control if the image should be binary, grayscale, rgb, etc.

        :param path: Full path to the image
        :param pil_format: The PIL format of the image to be loaded ('1':Binary, 'L':Grayscale, 'RGB':Color)
        :return: The image as tensor
        """
        img = Image.open(path)
        img = np.array(img.convert(pil_format), dtype=np.float32)
        img = tf.expand_dims(img, 2)

        # Downscale images that are too big
        height = img.shape[0]
        width = img.shape[1]
        if height or width >= 2000:
            height_scale = height // 1000
            width_scale = width // 1000
            scale = height_scale if height_scale > width_scale else width_scale
            height /= scale
            width /= scale

        height = self.closest_multiple(height)
        width = self.closest_multiple(width)
        img = tf.image.resize_with_pad(img, height, width)

        return img

    def __getitem__(self, index):
        """
        Indexing access to the Keras Sequence

        :param index: The index number of the image/label to be retrieved
        :return: The image as tensor and (possibly) label as tensor
        """
        img = self.tensor_image(os.path.join(self.img_path, self.imgs[index]), pil_format="L")
        img = tf.image.per_image_standardization(img)  # Adjust image to have mean 0 and variance 1

        # FOR TRAINING
        # If a label_path was given, convert the label to a tensor and return it along with the image tensor
        if self.label_path is not None:
            label = self.tensor_image(os.path.join(self.label_path, self.imgs[index]), pil_format="1")

            return img, label

        # FOR INFERENCE
        # If no label was given, return the image tensor and the image name
        return img, self.imgs[index].split('.')[0]

    def __len__(self):
        """
        The number of items in the sequence
        :return: Length of the sequence
        """
        return len(self.imgs)
