import sys

import yaml

import lineseg.dataset as ds
from lineseg.training import ModelTrainer
from lineseg.model import ARUNet

# Define the string names of all configuration arguments
TRAIN_CSV_PATH = 'train_csv_path'
VAL_CSV_PATH = 'val_csv_path'
SPLIT_TRAIN_SIZE = 'split_train_size'
MODEL_OUT = 'model_out'
MODEL_IN = 'model_in'
IMG_SIZE = 'img_size'
EPOCHS = 'epochs'
BATCH_SIZE = 'batch_size'
LEARNING_RATE = 'learning_rate'
SHUFFLE_SIZE = 'shuffle_size'


def train_model(cmd_args):
    """
    Train the model according to the parameters given

    python train.py <TRAIN_CONFIGURATION_FILE>

    Command Line Arguments:
    * train_configuration_file: The path to the train configuration file. An example config file is
                                given as "train_config.yaml"

    Configuration File Arguments:
    * train_csv_path: (Required) The path to the train images in the dataset
    * val_csv_path: (Optional) The path to the validation images in the dataset
    * split_train_size: (Required, if val_csv_path not set, else Optional) The ratio used to determine the size of the
    train/validation split. If split_train_size is set to 0.8, then the training set will contain 80% of the data, and
    validation 20%. The dataset is not shuffled before being split.
    * model_out: (Required) The path for where to store the model weights after training
    * model_in: (Required) The path to the pre-trained model weights
    * img_size: (Required) The height and width of the image after it has been resized
    * epochs: (Required) The number of epochs (times through the training set) to train
    * batch_size: (Required) The number of images in a mini-batch
    * learning_rate: (Required) The learning rate the optimizer uses during training
    * shuffle_size: (Required) The number of images that will be loaded into memory and shuffled during the training
    process. In most cases, this number shouldn't change. However, if you are running into memory constraints, you can
    lower this number. A shuffle_size of 0 results in no shuffling

    :param cmd_args: command line arguments
    :return: None
    """
    # Ensure the train config file is included
    if len(cmd_args) == 0:
        print('Must include path to train config file. The default file is includes as train_config.yaml')
        return

    # Read arguments from the config file
    with open(cmd_args[0]) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Create train/validation dataset depending on configuration settings
    # Split the train dataset depending on if the val_csv_path is empty
    if not configs[VAL_CSV_PATH]:  # Will evaluate to False is empty
        dataset_size = ds.get_dataset_size(configs[TRAIN_CSV_PATH])
        train_dataset_size = int(configs[SPLIT_TRAIN_SIZE] * dataset_size)
        val_dataset_size = dataset_size - train_dataset_size

        dataset = ds.get_encoded_dataset_from_csv(configs[TRAIN_CSV_PATH], eval(configs[IMG_SIZE]))

        train_dataset = dataset.take(train_dataset_size).map(ds.augment)
        if configs[SHUFFLE_SIZE] != 0:
            train_dataset = train_dataset.shuffle(configs[SHUFFLE_SIZE], reshuffle_each_iteration=True)
        train_dataset = train_dataset.batch(configs[BATCH_SIZE])

        val_dataset = dataset.skip(train_dataset_size)\
                             .batch(configs[BATCH_SIZE])
    else:  # Use the data as given in the train/validation csv files - no additional splits performed
        train_dataset_size = ds.get_dataset_size(configs[TRAIN_CSV_PATH])
        val_dataset_size = ds.get_dataset_size(configs[VAL_CSV_PATH])

        train_dataset = ds.get_encoded_dataset_from_csv(configs[TRAIN_CSV_PATH], eval(configs[IMG_SIZE]))\
            .map(ds.augment)
        if configs[SHUFFLE_SIZE] != 0:
            train_dataset = train_dataset.shuffle(configs[SHUFFLE_SIZE], reshuffle_each_iteration=True)
        train_dataset = train_dataset.batch(configs[BATCH_SIZE])

        val_dataset = ds.get_encoded_dataset_from_csv(configs[VAL_CSV_PATH], eval(configs[IMG_SIZE]))\
            .batch(configs[BATCH_SIZE])

    model = ARUNet()
    if configs[MODEL_IN]:
        model.load_weights(configs[MODEL_IN])

    # Create the trainer object and load in configuration settings
    trainer = ModelTrainer(model, configs[EPOCHS], configs[BATCH_SIZE], train_dataset, train_dataset_size, val_dataset,
                           val_dataset_size, configs[MODEL_OUT], lr=configs[LEARNING_RATE])

    # Train the model
    model, losses, ious = trainer.train()

    # Print the losses, intersection over union over the course of training
    print('Train Losses:', losses[0])
    print('Val Losses:', losses[1])
    print('Train IoU:', ious[0])
    print('Val IoU:', ious[1])


if __name__ == '__main__':
    train_model(sys.argv[1:])
