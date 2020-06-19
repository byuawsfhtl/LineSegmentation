import sys

from tensorflow import print as tfprint
from tensorflow.config import list_physical_devices
from tensorflow.data import TFRecordDataset
from matplotlib import pyplot as plt

from src.lineseg.dataset.sequence import ARUSequence
from src.lineseg.dataset.tfrecord import create_tfrecord_from_sequence, read_tfrecord
from src.lineseg.training import ModelTrainer
from src.lineseg.util.arguments import TArg, TrainArgParser


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


def train(cmd_args):
    """
    Train the model according to the parameters given

    python train.py --img_path <IMG_PATH> --label_path <IMG_LABEL_PATH> --model_out <WEIGHTS_OUT_PATH>
                    --img_resize <(HEIGHT, WIDTH) AS TUPLE> --epochs <NUM_EPOCHS> --batch_size <BATCH_SIZE>
                    --weights_path <WEIGHTS_IN_PATH> --learning_rate <LEARNING_RATE> --train_size <TRAIN_SET_SPLIT_SIZE>
                    --tfrecord_out <TFRECORD_OUT_PATH> --graphs <TRUE/FALSE>

    Command Line Arguments:
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

    :param cmd_args: command line arguments
    :return: None
    """

    # Parse the command line arguments so that they can be accessible from the newly created args object
    args = TrainArgParser(cmd_args)
    args.parse()

    # Let the user know if we have access to a GPU or not
    tfprint('Devices Available:', list_physical_devices())
    print('Devices Available:', list_physical_devices())

    # Create a Keras Sequence so that we can access data
    sequence = ARUSequence(args[TArg.IMG_PATH], args[TArg.LABEL_PATH],
                           tuple(args[TArg.IMG_DIM_AFTER_RESIZE]))

    # Create a tfrecord from the created sequence. This will speed up training dramatically
    create_tfrecord_from_sequence(sequence, args[TArg.TFRECORD_OUT_PATH])

    # Get the respective sizes for the dataset splits
    dataset_size = len(sequence)
    train_dataset_size = int(float(args[TArg.TRAIN_SIZE]) * dataset_size)
    val_dataset_size = dataset_size - train_dataset_size

    # Create a TFRecordDataset using the newly created tfrecord
    dataset = TFRecordDataset(args[TArg.TFRECORD_OUT_PATH])\
        .map(read_tfrecord)\
        .shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    train_dataset = dataset.take(train_dataset_size).batch(args[TArg.BATCH_SIZE])
    val_dataset = dataset.skip(val_dataset_size).batch(args[TArg.BATCH_SIZE])

    # Create the trainer object and load in configuration settings
    train = ModelTrainer(epochs=int(args[TArg.EPOCHS]), batch_size=int(args[TArg.BATCH_SIZE]),
                         train_dataset=train_dataset, train_dataset_size=train_dataset_size, val_dataset=val_dataset,
                         val_dataset_size=val_dataset_size, lr=float(args[TArg.LEARNING_RATE]),
                         weights_path=args[TArg.WEIGHTS_PATH])

    # Train the model
    model, losses, ious = train()

    # Save the Model weights to the specified path
    model.save_weights(args[TArg.MODEL_OUT])
    print('Model weights saved to', args[TArg.MODEL_OUT])

    # Print the losses, intersection over union over the course of training
    print('Train Losses:', losses[0])
    print('Val Losses:', losses[1])
    print('Train IoU:', ious[0])
    print('Val IoU:', ious[1])

    # Show the graphs if specified by the user
    if bool(args[TArg.SHOW_GRAPHS]):
        show_graph(losses[0], losses[1], 'Loss')
        show_graph(ious[0], ious[1], 'IoU')

    print('Finished Training.')


if __name__ == '__main__':
    train(sys.argv[1:])
