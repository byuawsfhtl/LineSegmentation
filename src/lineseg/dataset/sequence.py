import os
import numpy as np
from PIL import Image
import tensorflow as tf


class ARUSequence(tf.keras.utils.Sequence):
    """
    ARU-Sequence

    Keras Sequence class responsible for loading dataset in a TensorFlow compatible format.
    """
    def __init__(self, img_path, label_path=None, desired_size=(768, 1152)):
        """
        Set up paths and necessary variables.

        :param img_path: Path to the images
        :param label_path:
        :param desired_size:
        """
        self.img_path = img_path
        self.label_path = label_path

        if not os.path.exists(self.img_path):
            raise Exception('Images do not exist in', self.img_path)
        if self.label_path is not None and not os.path.exists(self.label_path):
            raise Exception('Labels do not exist in', self.label_path)

        self.desired_size = desired_size
        self.imgs = os.listdir(self.img_path)

    def resize(self, img, desired_size):
        """
        Method to resize the image to the desired size

        :param img: The image given as either PIL image or numpy array
        :param desired_size: Tuple designating the x, y value of the desired image size
        :return: The resized image as numpy array
        """
        img_size = np.array(img).shape

        img_ratio = img_size[0] / img_size[1]
        desired_ratio = desired_size[0] / desired_size[1]

        if img_ratio >= desired_ratio:
            # Solve by height
            new_height = desired_size[0]
            new_width = int(desired_size[0] // img_ratio)
        else:
            # Solve by width
            new_height = int(desired_size[1] * img_ratio)
            new_width = desired_size[1]

        img = np.array(img.resize((new_width, new_height)))

        border_top = desired_size[0] - new_height
        border_right = desired_size[1] - new_width

        img = np.pad(img, [(border_top, 0), (0, border_right)], mode='constant', constant_values=0)

        return img

    def tensor_image(self, path, pil_format):
        """
        Load an image from the given path, resize it, and convert it to a tensor. The PIL format is helpful
        to control if the image should be binary, grayscale, rgb, etc.

        :param path: Full path to the image
        :param pil_format: The PIL format of the image to be loaded ('1':Binary, 'L':Grayscale, 'RGB':Color)
        :return: The image as tensor
        """
        img = Image.open(path)
        img = img.convert(pil_format)
        img = self.resize(img, self.desired_size)
        x = tf.constant(img, dtype=tf.float32)

        return x

    def __getitem__(self, index):
        """
        Indexing access to the Keras Sequence

        :param index: The index number of the image/label to be retrieved
        :return: The image as tensor and (possibly) label as tensor
        """
        img = self.tensor_image(os.path.join(self.img_path, self.imgs[index]), pil_format="L")
        img = tf.expand_dims(img, 2)

        # If a label_path was given, convert the label to a tensor and return it along with the image tensor
        if self.label_path is not None:
            label = self.tensor_image(os.path.join(self.label_path, self.imgs[index].split('.')[0] + '_gt.jpg'),
                                      pil_format="1")
            label = tf.expand_dims(label, 2)

            return img, label

        # If no label was given, return the image tensor and the image name
        return img, self.imgs[index].split('.')[0]

    def __len__(self):
        """
        The number of items in the sequence
        :return: Length of the sequence
        """
        return len(self.imgs)
