import os
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import math

class Augmentor:
    def __init__(self, min_transforms=1, max_transforms=4):
        # Automatically collect all transformation methods from the class
        self.transforms = [getattr(self, method) for method in dir(self) if callable(getattr(self, method)) and method.startswith('random_')]
        self.min_transforms = min_transforms
        self.max_transforms = max_transforms
    
    def apply_transformations(self, image, gt_image):
        # Randomly choose how many transforms to apply (between min_transforms and max_transforms)
        num_transforms = random.randint(self.min_transforms, self.max_transforms)
        
        # Randomly select transforms from the list
        selected_transforms = random.sample(self.transforms, num_transforms)

        # Generate a random seed
        seed = random.randint(0, 2**31 - 1)
        tf.random.set_seed(seed)
        seed_tensor = tf.constant([seed, seed], dtype=tf.int32)
        
        for transform in selected_transforms:
            image, gt_image = transform(image, gt_image, seed_tensor)
        return image, gt_image

    def random_flip_left_right(self, image, gt_image, seed):
        image = tf.image.stateless_random_flip_left_right(image, seed=seed)
        gt_image = tf.image.stateless_random_flip_left_right(gt_image, seed=seed)
        return image, gt_image

    def random_flip_up_down(self, image, gt_image, seed):
        image = tf.image.stateless_random_flip_up_down(image, seed=seed)
        gt_image = tf.image.stateless_random_flip_up_down(gt_image, seed=seed)
        return image, gt_image

    def random_brightness(self, image, gt_image, seed):
        image = tf.image.stateless_random_brightness(image, max_delta=0.2, seed=seed)
        return image, gt_image

    def random_crop(self, image, gt_image, seed):
        image = tf.image.stateless_random_crop(value=image, size=(120, 180, 3), seed=seed)
        gt_image = tf.image.stateless_random_crop(value=gt_image, size=[120, 180, 3], seed=seed)
        return image, gt_image

    def random_blur(self, image, gt_image, seed):
        # Apply Gaussian blur only to the original image
        ksize = random.choice([(3, 3), (5, 5)])  # Choose a random kernel size
        image = tf.nn.conv2d(image[tf.newaxis, ...], tf.random.normal(ksize + (3, 3)), strides=[1, 1, 1, 1], padding='SAME')
        image = tf.squeeze(image, axis=0)  # Remove the added batch dimension
        return image, gt_image

    def random_img_quality(self, image, gt_image, seed):
        max_img_quality = random.randint(10, 90)
        min_img_quality = random.randint(0, max_img_quality)
        image = tf.image.stateless_random_jpeg_quality(image, min_img_quality, max_img_quality, seed)
        return image, gt_image

    def random_contrast(self, image, gt_image, seed):
        lower = random.uniform(0.5, 1.0)
        upper = random.uniform(1.0, 2.5)
        image = tf.image.stateless_random_contrast(image, lower, upper, seed)
        return image, gt_image

    def random_rotation(self, image, gt_image=None, seed=None):
        # Define the 8 possible transformations using k values for rot90 and flip options
        transformations = [
            (0, False, False),  # 0 degrees, no flip
            (1, False, False),  # 90 degrees
            (2, False, False),  # 180 degrees
            (3, False, False),  # 270 degrees
            (0, True, False),   # 0 degrees, horizontal flip
            (1, True, False),   # 90 degrees, horizontal flip
            (2, True, False),   # 180 degrees, horizontal flip
            (3, True, False)    # 270 degrees, horizontal flip
        ]

        # Randomly select one of the transformations
        k, flip_horizontal, flip_vertical = random.choice(transformations)

        # Apply the rotation
        image = tf.image.rot90(image, k=k)
        gt_image = tf.image.rot90(gt_image, k=k)

        # Apply the flips if specified
        if flip_horizontal:
            image = tf.image.flip_left_right(image)
            gt_image = tf.image.flip_left_right(gt_image)

        if flip_vertical:
            image = tf.image.flip_up_down(image)
            gt_image = tf.image.flip_up_down(gt_image)

        return image, gt_image

    def distort_bboxes(self, image, gt_image, seed):
        pass  # Placeholder for bounding box distortion
    