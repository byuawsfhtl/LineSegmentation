import os
import random

import numpy as np
from PIL import Image
import tensorflow as tf


def random_augmentation(img, label):
    """
    Function to apply random augmentations to the image and label image

    :param img: The image to be transformed
    :param label: The label to be transformed
    :return: The transformed image and label
    """
    theta = 0
    tx = 0
    ty = 0
    zx = 1
    zy = 1
    shear = 0

    # Random Flip
    if random.randint(0, 2) == 0:
        img = tf.image.flip_left_right(img)
        label = tf.image.flip_left_right(label)
    # Random Rotate
    if random.randint(0, 1):  # .5
        theta = random.uniform(-2, 2)
    # Random Shear
    elif random.randint(0, 1):  # Only do shear if we haven't rotated
        shear = random.uniform(-2, 2)
    # Random Zoom
    if random.randint(0, 2) == 0:
        zx = random.uniform(0.9, 1.1)
        zy = random.uniform(0.9, 1.1)
    # Random Translation
    if random.randint(0, 1):
        tx = random.uniform(-10, 10)
        ty = random.uniform(-10, 10)

    # Apply Affine Transformation
    img = tf.keras.preprocessing.image.apply_affine_transform(img.numpy(), theta=theta, tx=tx, ty=ty, shear=shear,
                                                              zx=zx, zy=zy)
    label = tf.keras.preprocessing.image.apply_affine_transform(label.numpy(), theta=theta, tx=tx, ty=ty, shear=shear,
                                                                zx=zx, zy=zy)

    # Apply Random Brightness Transformation
    # if random.randint(0, 1):
    #     img = tf.keras.preprocessing.image.random_brightness(img, (.01, 1.4))
    # Apply Random Channel Shift
    # else:
    #     img = tf.keras.preprocessing.image.random_channel_shift(img, 100)

    return img, label


class ARUSequence(tf.keras.utils.Sequence):
    """
    ARU-Sequence

    Keras Sequence class responsible for loading dataset in a TensorFlow compatible format.
    """
    def __init__(self, img_path, label_path=None, desired_size=(768, 1152), augmentation_rate=1):
        """
        Set up paths and necessary variables.

        :param img_path: Path to the images
        :param label_path:
        :param desired_size:
        """
        self.img_path = img_path
        self.label_path = label_path
        self.augmentation_rate = augmentation_rate

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
        img_index = index // self.augmentation_rate

        img = self.tensor_image(os.path.join(self.img_path, self.imgs[img_index]), pil_format="L")
        img = tf.expand_dims(img, 2)

        # FOR TRAINING
        # If a label_path was given, convert the label to a tensor and return it along with the image tensor
        if self.label_path is not None:
            label = self.tensor_image(os.path.join(self.label_path, self.imgs[img_index].split('.')[0] + '_gt.jpg'),
                                      pil_format="1")
            label = tf.expand_dims(label, 2)

            # Don't perform data augmentation the first time we see this image
            if index % self.augmentation_rate != 0:
                img, label = random_augmentation(img, label)

            return img, label

        # FOR INFERENCE
        # If no label was given, return the image tensor and the image name
        return img, self.imgs[index].split('.')[0]

    def __len__(self):
        """
        The number of items in the sequence
        :return: Length of the sequence
        """
        return len(self.imgs) * self.augmentation_rate
