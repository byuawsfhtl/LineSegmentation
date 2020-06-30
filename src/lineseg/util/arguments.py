from enum import Enum
from abc import ABC, abstractmethod


class IArg(Enum):
    """
    Inference Arguments
    """
    IMG_PATH = '--img_path'
    OUT_PATH = '--out_path'
    WEIGHTS_PATH_BASELINE = '--weights_path_baseline'
    WEIGHTS_PATH_SEAM = '--weights_path_seam'
    SEGMENTATION_STEP_SIZE = '--seg_step_size'
    SHOULD_PLOT_IMAGES = '--plot'
    IMAGE_DIM_AFTER_RESIZE = '--image_resize'


class TArg(Enum):
    """
    Train Arguments
    """
    IMG_PATH = '--img_path'
    LABEL_PATH = '--label_path'
    MODEL_OUT = '--model_out'
    IMG_DIM_AFTER_RESIZE = '--img_resize'
    EPOCHS = '--epochs'
    BATCH_SIZE = '--batch_size'
    WEIGHTS_PATH = '--weights_path'
    LEARNING_RATE = '--learning_rate'
    TRAIN_SIZE = '--train_size'
    TFRECORD_IN_PATH = '--tfrecord_in'
    TFRECORD_OUT_PATH = '--tfrecord_out'
    SHOW_GRAPHS = '--graphs'
    SAVE_BEST_AFTER = '--save_best_after'
    AUGMENTATION_RATE = '--augmentation_rate'


class ArgParser(ABC):  # Abstract Class
    """
    Base class for argument parsers.

    Any subclass of ArgParser needs to override the *parse* method, set the default
    parameters and call the *add_arguments* and *check_required_args* methods.
    """
    def __init__(self, args, arg_type):
        self.args = args
        self.arg_type = arg_type
        self.arg_dict = dict()
        self.REQUIRED_PARAM_MESSAGE = 'The {} argument must be specified.'

    @abstractmethod
    def parse(self):
        """
        Abstract method that must be overridden to subclass ArgumentParser

        :return: None
        """
        pass

    def get(self, arg):
        """
        Given an Enum Arg, return the value

        :param arg: Enum Arg
        :return: Value of Enum Arg
        """
        if type(arg) is not self.arg_type:
            raise Exception('Argument must be of type ' + str(self.arg_type))

        return self.arg_dict.get(arg.value)  # Use get method to ensure we return None if dict is empty

    def check_required_args(self, required_args):
        """
        Given a list of Enum Arg values, check to make sure these values are present in our argument dictionary

        :param required_args: List of Enum Arg values
        :return: None
        """
        for arg in required_args:
            if arg.value not in self.arg_dict:
                raise Exception(self.REQUIRED_PARAM_MESSAGE.format(arg.value))

    def add_arguments(self):
        """
        Iterates over command line arguments and adds them to the argument dictionary

        :return: None
        """
        index = 0
        while index < len(self.args):
            if self.args[index] in [e.value for e in self.arg_type]:
                self.arg_dict[self.args[index]] = self.args[index + 1]
                index += 2
            else:
                raise Exception('Unexpected command line argument ' + self.args[index])

    def __getitem__(self, arg):
        return self.get(arg)


class TrainArgParser(ArgParser):
    """
    Argument Parser for Train Arguments
    """
    def __init__(self, args):
        super(TrainArgParser, self).__init__(args, TArg)

    def parse(self):
        """
        Function to parse command line arguments for train

        :return: dictionary with configuration settings
        """
        self.arg_dict[TArg.IMG_DIM_AFTER_RESIZE.value] = '(1024, 1536)'
        self.arg_dict[TArg.EPOCHS.value] = '100'
        self.arg_dict[TArg.BATCH_SIZE.value] = '1'
        self.arg_dict[TArg.LEARNING_RATE.value] = '1e-3'
        self.arg_dict[TArg.TRAIN_SIZE.value] = '0.8'
        self.arg_dict[TArg.TFRECORD_OUT_PATH.value] = './data/misc/data.tfrecord'
        self.arg_dict[TArg.SHOW_GRAPHS.value] = 'False'
        self.arg_dict[TArg.SAVE_BEST_AFTER.value] = '10'
        self.arg_dict[TArg.AUGMENTATION_RATE.value] = '20'

        # Add Arguments to arg_dict and ensure required args are present
        self.add_arguments()
        self.check_required_args([TArg.IMG_PATH, TArg.LABEL_PATH, TArg.MODEL_OUT])


class InfArgParser(ArgParser):
    """
    Argument Parser for Inference Arguments
    """
    def __init__(self, args):
        super(InfArgParser, self).__init__(args, IArg)

    def parse(self):
        """
        Function to parse command line arguments for inference

        :return: dictionary with configuration settings
        """
        # Set Default Arguments
        self.arg_dict[IArg.SHOULD_PLOT_IMAGES.value] = 'False'
        self.arg_dict[IArg.SEGMENTATION_STEP_SIZE.value] = '1'
        self.arg_dict[IArg.IMAGE_DIM_AFTER_RESIZE.value] = "(1024, 1536)"

        # Add Arguments to arg_dict and ensure required args are present
        self.add_arguments()
        self.check_required_args([IArg.IMG_PATH, IArg.OUT_PATH, IArg.WEIGHTS_PATH_BASELINE, IArg.WEIGHTS_PATH_SEAM])
