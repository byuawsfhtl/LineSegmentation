class ArgumentParser:
    def __init__(self, args):
        self.args = args

        # ARGUMENTS
        self.ARG_IDENT = '--'
        self.IMG_PATH = 'img_path'
        self.OUT_PATH = 'out_path'
        self.WEIGHTS_PATH_BASELINE = 'weights_path_baseline'
        self.WEIGHTS_PATH_SEAM = 'weights_path_seam'
        self.SEGMENTATION_STEP_SIZE = 'seg_step_size'
        self.SHOULD_PLOT_IMAGES = 'plot'
        self.IMAGE_DIM_AFTER_RESIZE = 'image_resize'
        self.REQUIRED_PARAM_MESSAGE = 'The {} argument must be specified.'

    def parse_inference_arguments(self):
        """
        Function to parse command line arguments for inference

        :return: dictionary with configuration settings
        """
        arg_dict = dict()

        index = 0
        while index < len(self.args):
            if self.args[index] == self.ARG_IDENT + self.IMG_PATH:
                index += 1
                arg_dict[self.IMG_PATH] = self.args[index]
            elif self.args[index] == self.ARG_IDENT + self.OUT_PATH:
                index += 1
                arg_dict[self.OUT_PATH] = self.args[index]
            elif self.args[index] == self.ARG_IDENT + self.WEIGHTS_PATH_BASELINE:
                index += 1
                arg_dict[self.WEIGHTS_PATH_BASELINE] = self.args[index]
            elif self.args[index] == self.ARG_IDENT + self.WEIGHTS_PATH_SEAM:
                index += 1
                arg_dict[self.WEIGHTS_PATH_SEAM] = self.args[index]
            elif self.args[index] == self.ARG_IDENT + self.SEGMENTATION_STEP_SIZE:
                index += 1
                arg_dict[self.SEGMENTATION_STEP_SIZE] = self.args[index]
            elif self.args[index] == self.ARG_IDENT + self.SHOULD_PLOT_IMAGES:
                arg_dict[self.SHOULD_PLOT_IMAGES] = True
            elif self.args[index] == self.ARG_IDENT + self.IMAGE_DIM_AFTER_RESIZE:
                index += 1
                arg_dict[self.IMAGE_DIM_AFTER_RESIZE] = self.args[index]
            else:
                raise Exception('Unexpected command line argument:' + self.args[index])

            index += 1

        # Required command line arguments
        if self.IMG_PATH not in arg_dict:
            raise Exception(self.REQUIRED_PARAM_MESSAGE.format(self.IMG_PATH))
        if self.OUT_PATH not in arg_dict:
            raise Exception(self.REQUIRED_PARAM_MESSAGE.format(self.OUT_PATH))
        if self.WEIGHTS_PATH_BASELINE not in arg_dict:
            raise Exception(self.REQUIRED_PARAM_MESSAGE.format(self.WEIGHTS_PATH_BASELINE))
        if self.WEIGHTS_PATH_SEAM not in arg_dict:
            raise Exception(self.REQUIRED_PARAM_MESSAGE.format(self.WEIGHTS_PATH_SEAM))

        # Default arguments
        if self.SHOULD_PLOT_IMAGES not in arg_dict:
            arg_dict[self.SHOULD_PLOT_IMAGES] = False
        if self.SEGMENTATION_STEP_SIZE not in arg_dict:
            arg_dict[self.SEGMENTATION_STEP_SIZE] = '1'
        if self.IMAGE_DIM_AFTER_RESIZE not in arg_dict:
            arg_dict[self.IMAGE_DIM_AFTER_RESIZE] = "(768, 1152)"

        return arg_dict


def parse_train_arguments(args):
    """
    Function to parse command line arguments for training

    :param args: command line arguments passed to the train python file
    :return: dictionary with configuration settings
    """
    arg_dict = dict()

    index = 0
    while index < len(args):
        if args[index] == '--img_path':
            index += 1
            arg_dict['img_path'] = args[index]
        elif args[index] == '--label_path':
            index += 1
            arg_dict['label_path'] = args[index]
        elif args[index] == '--show_graphs':
            arg_dict['show_graphs'] = True
        elif args[index] == '--log_level':
            index += 1
            arg_dict['log_level'] = args[index]
        elif args[index] == '--model_out':
            index += 1
            arg_dict['model_out'] = args[index]
        elif args[index] == '--epochs':
            index += 1
            arg_dict['epochs'] = int(args[index])
        elif args[index] == '--batch_size':
            index += 1
            arg_dict['batch_size'] = int(args[index])
        elif args[index] == '--learning_rate':
            index += 1
            arg_dict['learning_rate'] = float(args[index])
        elif args[index] == '--max_seq_size':
            index += 1
            arg_dict['max_seq_size'] = int(args[index])
        elif args[index] == '--train_size':
            index += 1
            arg_dict['train_size'] = float(args[index])
        elif args[index] == '--tfrecord_out':
            index += 1
            arg_dict['tfrecord_out'] = args[index]
        elif args[index] == '--weights_path':
            index += 1
            arg_dict['weights_path'] = args[index]
        elif args[index] == '--metrics':
            arg_dict['metrics'] = True
        else:
            raise Exception('Unexpected command line argument:' + args[index])

        index += 1

    # Required command line arguments
    if 'img_path' not in arg_dict:
        raise Exception('The img_path argument must be set!')
    if 'label_path' not in arg_dict:
        raise Exception('The label_path argument must be set!')

    # Set arguments to their defaults if not present on command line
    if 'show_graphs' not in arg_dict:
        arg_dict['show_graphs'] = False
    if 'log_level' not in arg_dict:
        arg_dict['log_level'] = '3'
    if 'model_out' not in arg_dict:
        arg_dict['model_out'] = './data/model_weights/hwr_model/run1'
    if 'epochs' not in arg_dict:
        arg_dict['epochs'] = 100
    if 'batch_size' not in arg_dict:
        arg_dict['batch_size'] = 100
    if 'learning_rate' not in arg_dict:
        arg_dict['learning_rate'] = 4e-4
    if 'max_seq_size' not in arg_dict:
        arg_dict['max_seq_size'] = 128
    if 'train_size' not in arg_dict:
        arg_dict['train_size'] = .8
    if 'tfrecord_out' not in arg_dict:
        arg_dict['tfrecord_out'] = './data/misc/data.tfrecords'
    if 'weights_path' not in arg_dict:
        arg_dict['weights_path'] = None
    if 'metrics' not in arg_dict:
        arg_dict['metrics'] = False

    return arg_dict
