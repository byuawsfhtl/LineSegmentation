# ARUNetSegmentation

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

## Usage

### Train
TRAINING DOCUMENTATION AND SCRIPTS COMING SOON!

Training can be run with the following command:

`
python train.py ...
`


### Inference

Using the ```inference.py``` script, you can perform inference on a
pre-trained model.

Inference can be performed by running the following command:
`
python inference.py --img_path <PATH_TO_IMAGES> --out_path <PATH_TO_OUTPUT_SNIPPETS> --weights_path_baseline <PATH_TO_BASELINE_MODEL_WEIGHTS> --weights_path_seam <PATH_TO_SEAM_MODEL_WEIGHTS>
`

Optionally, a number of command line arguments can be used to alter inference behavior.
The full list of parameters include:
* img_path (required): The path to the images to be inferred
* out_path (required): The path to the results of the inference (text-line snippets)
* weights_path_baseline (required): The path to the pre-trained model weights for baselines
* weights_path_seam (required): The path to the pre-trained model weights for seams
* seg_step_size (optional): How many columns along the baseline to look at when searching the seam image to find
                            the bounding polygon (default: 1)
* plot (optional): Should each text line snippet be shown to the screen during inference? (default: False)
* image_resize (optional): The height and width for resizing the image when sent into the model for inference
                           (default: 768, 1152)
