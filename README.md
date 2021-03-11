# LineSegmentation

This project contains code necessary to perform line-level segmentation
in TensorFlow 2. Using the provided scripts, the model can be trained and
also used for inference.

This project can be used by cloning the repository and running the manually.
However, it is also available in [Anaconda Cloud](https://anaconda.org/BYU-Handwriting-Lab/lineseg)
and can be used in any Conda environment.

## Dependencies
* Python 3.x
* TensorFlow 2.x
* Numpy
* Pillow
* Pandas
* Matplotlib
* Tqdm
* Scikit-learn

A .yaml file for each supported platform has been included that specifies the necessary dependencies. A
conda environment for MacOS/Windows/Linux can be created and activated by running the following commands:

```
conda env create -f environment_linux.yaml  # or environment_macos.yaml, environment_windows.yaml
conda activate lineseg_env
```

## Usage with Provided Scripts

Using the code available in this repository, you have access to the ```train.py```
and ```inference.py``` scripts.

### Train

Training can be done with the following command:

```
python train.py <TRAIN_CONFIG_FILE>
```

The train configuration file contains all the settings needed to train a line segmentation model.
To train your own model, simply modify the configuration file arguments. Explanations of the
arguments are given below:

Configuration File Arguments:
* train_csv_path: The path to the train images in the dataset
* val_csv_path: The path to the validation images in the dataset. If this parameter is not set, the training set will be
                split according to the train_size parameter.
* train_size: The ratio used to determine the size of the train/validation split. If split_train_size is set to 0.8,
then the training set will contain 80% of the data, and validation 20%. The dataset is not shuffled before being split.
* model_out: The path for where to store the model weights after training
* model_in: The path to the pre-trained model weights
* img_size: The height and width of the image after it has been resized
* epochs: The number of epochs (times through the training set) to train
* batch_size: The number of images in a mini-batch
* learning_rate: The learning rate the optimizer uses during training
* shuffle_size: The number of images that will be loaded into memory and shuffled during the training process In most
                cases, this number shouldn't change. However, if you are running into memory constraints, you can lower
                this number. A shuffle_size of 0 results in no shuffling

### Inference

Using the ```inference.py``` script, you can perform inference on a
pre-trained model.

Inference can be performed by running the following command:

```
python inference.py <INFERENCE_CONFIG_FILE>
```

The inference configuration file contains all the settings needed to perform inference on a line segmentation model.
To perform inference on your own model, simply modify the configuration file arguments. Explanations of the
arguments are given below:

Configuration File Arguments:
* img_path: The path to the directory of images to be inferred
* img_path_subdirs: Whether or not to include images included in subdirectories of the img_path
* out_path: The path to the directory that segmented line snippets will be stored
* model_in: The path to the pre-trained model weights
* coordinate_naming: Whether or not to save the coordinate information in each line snippet's name
* save_raw: Whether or not to save the raw output of the semantic segmentation model
* raw_path: The path to the directory that the raw output images will be stored
* img_size: The size which all images will be resized for inference
* batch_size: The size of the mini-batch used during inference

## Usage with Conda Package

Potentially, the easiest way to access the code is to import the [conda package](https://anaconda.org/byu-handwriting-lab/lineseg)
that is available on Anaconda-Cloud. No cloning of this repository is necessary.

```
conda install -c byu-handwriting-lab lineseg
```

Code can then be accessed like any normal python package. For example, to use the recognition model, you could write
something like this:

```
from lineseg.model import ARUNet
from lineseg.seg import segment_from_predictions

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = ARUNet()

# Load some pretrained weights
model_weights_path = './some/path/to/model/weights'
model.load_weights(model_weights_path)

# Simulate creating an image with random numbers
path_to_images = '/path/to/images/to/be/inferred/'
output_path = 'path/to/save/text/line/snippets'
img_size = (1024, 1536)

dataset = ds.get_encoded_inference_dataset_from_img_path(path_to_images, img_size)

# Run the images through the segmentation model
for image, img_name in dataset:
    output = model(image)
    prediction = tf.argmax(output, axis=3)

    # Show the raw model output
    plt.imshow(tf.squeeze(prediction))
    plt.pause(0.01)

    # Segment individual lines based on model output
    segment_from_predictions(image, prediction, img_name, output_path, plot_images=True)
```

## Build the Conda Package to be uploaded to Anaconda Cloud

This project can be packaged with Anaconda and uploaded to the cloud. It is done through the use of ```setup.py```
and ```meta.yaml```. Slight modifications to these files may need to take place if dependencies to the code base change.
The project can be packaged using the following ```conda-build``` command.

```
conda-build ./conda.recipe -c defaults -c conda-forge
```

Make sure the lineseg environment has been built and activated before you run the conda-build command.

```
conda env create -f environment.yaml
conda activate linseg_env
```

Once the project has been packaged, the packaged file can be uploaded to Anaconda Cloud (Anaconda-Client is required):

```
anaconda upload -u BYU-Handwriting-Lab <FILENAME>
```
