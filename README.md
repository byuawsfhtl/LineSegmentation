# LineSegmentation

This project contains code necessary to perform line-level segmentation
in TensorFlow 2. Using the provided scripts, the model can be trained and
also used for inference.

## Dependencies
* TensorFlow 2.x
* Python 3.x
* Numpy
* Pillow
* Matplotlib
* Tqdm
* Shapely
* Scikit-learn

A .yaml file has been included that specifies the necessary dependencies. A
conda environment can be created and activated by running the following
commands:

`
conda env create -f environment.yaml
conda activate lineseg_env
`

## Pre-trained Weights

Pre-trained weights can be found on the BYU Handwriting Lab's Google Drive. Look in the Google Account repo for
access credentials. The weights can be found under the following paths: ```arunet_baselines``` and ```arunet_seams```

The images that the weights were trained on are also found on the Google Drive and can be found here:
```baseline_seams_dataset.zip```

## Usage

### Train

Training can be run with the following command:

`
python train.py --img_path <PATH_TO_IMAGES> --label_path <PATH_TO_GROUND_TRUTH_IMAGES> --model_out <MODEL_WEIGHTS_OUT_PATH>
`

Optionally, a number of command line arguments can be specified to alter training behavior:

A full list of the arguments include:

* img_path (required): The path to the images in the dataset
* label_path (required): The path to the ground truth image labels in the dataset
* model_out (required): The path to store the model weights
* img_resize (optional): The height and width of the image after it has been resized (default: (768, 1152)
* epochs (optional): The number of epochs to train (default: 100)
* batch_size (optional): The number of images in a mini-batch (default:2)
* weights_path (optional): The path to the pre-trained model weights (default: None)
* learning_rate (optional): The learning rate the optimizer uses during training (default: 1e-3)
* train_size (optional): The ratio used to determine the size of the train/validation sets (default: 0.8)
* tfrecord_out (optional): The path to the created tfrecords file (default: ./data/misc/data.tfrecords)
* graphs (optional): Whether or not to show graphs of the loss/IoU after training (default: False)
* save_best_after (optional): How many epochs will pass before the model weights are saved (if it has achieved the
                              the best accuracy on the validation set) during the training process (default: 25)
* augmentation_rate (optional): The rate of extra images that will be applied to the dataset during training. A
                                rate of 1 means no data augmentation (default: 20)

### Inference

Using the ```inference.py``` script, you can perform inference on a
pre-trained model.

Inference can be performed by running the following command:

`
python inference.py --img_path <PATH_TO_IMAGES> --out_path <PATH_TO_OUTPUT_SNIPPETS> --weights_path_baseline <PATH_TO_BASELINE_MODEL_WEIGHTS> --weights_path_seam <PATH_TO_SEAM_MODEL_WEIGHTS>
`

Optionally, a number of command line arguments can be used to alter inference behavior.

The full list of arguments include:

* img_path (required): The path to the images to be inferred
* out_path (required): The path to the results of the inference (text-line snippets)
* weights_path_baseline (required): The path to the pre-trained model weights for baselines
* weights_path_seam (required): The path to the pre-trained model weights for seams
* seg_step_size (optional): How many columns along the baseline to look at when searching the seam image to find
                            the bounding polygon (default: 1)
* plot (optional): Should each text line snippet be shown to the screen during inference? (default: False)
* image_resize (optional): The height and width for resizing the image when sent into the model for inference
                           (default: 768, 1152)
