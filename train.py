import sys

import tensorflow as tf
from matplotlib import pyplot as plt
import yaml

import lineseg.dataset as ds
from lineseg.training import ModelTrainer

# Define the string names of all configuration arguments
TRAIN_CSV_PATH = 'train_csv_path'
VAL_CSV_PATH = 'val_csv_path'
SPLIT_TRAIN = 'split_train'
TRAIN_SIZE = 'train_size'
MODEL_OUT = 'model_out'
MODEL_IN = 'model_in'
IMG_SIZE = 'img_size'
EPOCHS = 'epochs'
BATCH_SIZE = 'batch_size'
LEARNING_RATE = 'learning_rate'
SHOW_GRAPHS = 'show_graphs'
SAVE_EVERY = 'save_every'
SHUFFLE_SIZE = 'shuffle_size'


def train_model(cmd_args):
    """
    Train the model according to the parameters given

    python train.py <TRAIN_CONFIGURATION_FILE>

    Command Line Arguments:
    * train_configuration_file: The path to the train configuration file. An example config file is
                                given as "train_config.yaml"

    Configuration File Arguments:
    * img_path (required): The path to the images in the dataset
    * label_path (required): The path to the ground truth image labels in the dataset
    * model_out (required): The path to store the model weights
    * img_resize (optional): The height and width of the image after it has been resized (default: (768, 1152)
    * epochs (optional): The number of epochs to train (default: 100)
    * batch_size (optional): The number of images in a mini-batch (default:2)
    * weights_path (optional): The path to the pre-trained model weights (default: None)
    * learning_rate (optional): The learning rate the optimizer uses during training (default: 1e-3)
    * train_size (optional): The ratio used to determine the size of the train/validation sets (default: 0.8)
    * tfrecord_in (optional): The path to a previously created tfrecords file. This argument can be specified to skip
                              the creation of a tfrecord during training (default: None)
    * tfrecord_out (optional): The path to the created tfrecords file (default: ./data/misc/data.tfrecords)
    * graphs (optional): Whether or not to show graphs of the loss/IoU after training (default: False)
    * save_best_after (optional): How many epochs will pass before the model weights are saved (if it has achieved the
                                  the best accuracy on the validation set) during the training process (default: 10)
    * augmentation_rate (optional): The rate of extra images that will be applied to the dataset during training. A
                                    rate of 1 means no data augmentation (default: 20)

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

    # Print available devices so we know if we are using CPU or GPU
    tf.print('Devices Available:', tf.config.list_physical_devices())

    # Create train/validation dataset depending on configuration settings
    # Split the train dataset based on the TRAIN_SIZE parameter
    if configs[SPLIT_TRAIN]:
        dataset_size = ds.get_dataset_size(configs[TRAIN_CSV_PATH])
        train_dataset_size = int(configs[TRAIN_SIZE] * dataset_size)
        val_dataset_size = dataset_size - train_dataset_size

        dataset = ds.get_encoded_dataset_from_csv(configs[TRAIN_CSV_PATH], eval(configs[IMG_SIZE]))
        train_dataset = dataset.take(train_dataset_size)\
                               .shuffle(configs[SHUFFLE_SIZE], reshuffle_each_iteration=True)\
                               .batch(configs[BATCH_SIZE])
        val_dataset = dataset.take(val_dataset_size)\
                             .batch(configs[BATCH_SIZE])
    else:  # Use the data as given in the train/validation csv files - no additional splits performed
        train_dataset_size = ds.get_dataset_size(configs[TRAIN_CSV_PATH])
        val_dataset_size = ds.get_dataset_size(configs[VAL_CSV_PATH])

        train_dataset = ds.get_encoded_dataset_from_csv(configs[TRAIN_CSV_PATH], eval(configs[IMG_SIZE]))\
            .shuffle(configs[SHUFFLE_SIZE])\
            .batch(configs[BATCH_SIZE])
        val_dataset = ds.get_encoded_dataset_from_csv(configs[VAL_CSV_PATH], eval(configs[IMG_SIZE]))\
            .batch(configs[BATCH_SIZE])

    # Create the trainer object and load in configuration settings
    trainer = ModelTrainer(configs[EPOCHS], configs[BATCH_SIZE], train_dataset, train_dataset_size, val_dataset,
                           val_dataset_size, configs[MODEL_OUT], model_in=configs[MODEL_IN], lr=configs[LEARNING_RATE],
                           save_every=configs[SAVE_EVERY])

    # Train the model
    model, losses, ious = trainer.train()

    # Print the losses, intersection over union over the course of training
    print('Train Losses:', losses[0])
    print('Val Losses:', losses[1])
    print('Train IoU:', ious[0])
    print('Val IoU:', ious[1])

    # Show the graphs if specified by the user
    if configs[SHOW_GRAPHS]:
        show_graph(losses[0], losses[1], 'Loss')
        show_graph(ious[0], ious[1], 'IoU')


def show_graph(train_metric, val_metric, title):
    """
    Creates a graph showing a given metric over time

    :param train_metric:  list of the given metric on the training set per epoch
    :param val_metric: list of the given metric on the validation set per epoch
    :param title: the metric type which will be used for the title and y-axis label
    :return: None
    """
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(title)
    plt.plot(train_metric, label='Train')
    plt.plot(val_metric, label='Val')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_model(sys.argv[1:])
