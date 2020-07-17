import tensorflow as tf
from tqdm import tqdm

from src.lineseg.model import ARUNet
from src.lineseg.dataset.tfrecord import read_tfrecord


def subsample(img, label):
    index = tf.random.uniform([], 0, 4, dtype=tf.int32)

    height = tf.shape(img)[0]
    width = tf.shape(img)[1]

    if index == 0:
        img = tf.image.resize(img, (height // 2, width // 2))
        label = tf.image.resize(label, (height // 2, width // 2))
    elif index == 1:
        img = tf.image.resize(img, (height // 2, width))
        label = tf.image.resize(label, (height // 2, width))
    elif index == 2:
        img = tf.image.resize(img, (height, width // 2))
        label = tf.image.resize(label, (height, width // 2))

    return img, label


def augment(img, label):
    mix = tf.concat((img, label), axis=2)

    if tf.random.uniform((), 0, 2, dtype=tf.int32) != 0:
        crop_height = tf.random.uniform((), 768, 1025, dtype=tf.int32)
        crop_width = tf.random.uniform((), 1024, 1280, dtype=tf.int32)

        height = int(tf.shape(img)[0])
        width = int(tf.shape(img)[1])
        mix = tf.image.random_crop(mix, (crop_height, crop_width, 2))
        mix = tf.image.resize_with_pad(mix, height, width)

    mix = tf.image.random_flip_left_right(mix)

    img = tf.expand_dims(mix[:, :, 0], 2)
    label = tf.expand_dims(mix[:, :, 1], 2)

    return img, label


class ModelTrainer:
    """
    ModelTrainer

    Responsible for training the model. Scope becomes an issues when dealing with @tf.function.
    It's easier to place all of the training code into an object so we don't run into issues.
    Once the object is created, the __call__ method will train and return the results and the
    trained model.
    """

    def __init__(self, epochs, batch_size, dataset_path, train_dataset_size, val_dataset_size, save_path,
                 lr=1e-3, weights_path=None, save_best_after=25, shuffle_size=10):
        """
        Set up the necessary variables that will be used during training, including the model, optimizer,
        encoder, and other metrics.

        :param epochs: The number of epochs to train the model
        :param batch_size: How many images will be included in a mini-batch
        :param dataset_path: The path to the TfRecord dataset
        :param train_dataset_size: The size of the train dataset
        :param val_dataset_size: The size of the val dataset
        :param lr: The learning rate
        :param weights_path: The path to the weights if we are starting from a pre-trained model
        :param save_best_after: Save model weights (if it achieved best IoU) after how many epochs?
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_dataset_size = train_dataset_size
        self.val_dataset_size = val_dataset_size
        self.save_path = save_path
        self.save_best_after = save_best_after

        self.dataset = tf.data.TFRecordDataset(dataset_path).map(read_tfrecord)
        self.train_dataset = self.dataset.take(train_dataset_size).shuffle(shuffle_size, reshuffle_each_iteration=True)\
                                         .map(subsample).map(augment).batch(self.batch_size)
        self.val_dataset = self.dataset.skip(train_dataset_size).batch(self.batch_size)

        self.model = ARUNet()
        if weights_path is not None:
            self.model.load_weights(weights_path)

        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        self.objective = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')

        self.train_iou = tf.keras.metrics.MeanIoU(num_classes=2, name='train_iou')
        self.val_iou = tf.keras.metrics.MeanIoU(num_classes=2, name='val_iou')

    @tf.function
    def train_step(self, images, labels):
        """
        One training step given a mini-batch of images and labels. Note the use of the annotation, @tf.function.
        This annotation will allow TensorFlow to analyze the method and speed up training. However you must be
        careful on what is placed inside @tf.function. See the following links for details:
        * https://www.tensorflow.org/api_docs/python/tf/function
        * https://www.tensorflow.org/guide/function
        * https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/
        * https://pgaleone.eu/tensorflow/tf.function/2019/04/03/dissecting-tf-function-part-2/

        :param images: mini-batch of images in tensor format
        :param labels: mini-batch of labels in tensor format
        :return: None
        """
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.objective(tf.one_hot(tf.cast(labels, tf.int32), 2), predictions)
            loss += tf.add_n(self.model.losses)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_iou(labels, tf.argmax(predictions, axis=3))

    @tf.function
    def val_step(self, images, labels):
        """
        One validation step given a mini-batch of images and labels

        :param images: mini-batch of images in tensor format
        :param labels: mini-batch of labels in tensor format
        :return: None
        """
        predictions = self.model(images, training=False)
        loss = self.objective(tf.one_hot(tf.cast(labels, tf.int32), 2), predictions)
        loss += tf.add_n(self.model.losses)

        self.val_loss(loss)
        self.val_iou(labels, tf.argmax(predictions, axis=3))

    def __call__(self):
        """
        Main Training Loop

        This method trains the model according to the parameters passed in __init__. It will run for the specified
        number of epochs, keep track of loss and IoU metrics and return the model and the metrics.

        :return: Trained Model, (Train-Loss over time, Validation-Loss over time),
                 (Train-IoU over time, Validation-IoU over time)
        """
        best_val_iou = 0.0

        train_losses, val_losses = [], []
        train_ious, val_ious = [], []

        try:
            for epoch in range(self.epochs):
                # Reset our metrics for each epoch
                self.train_loss.reset_states()
                self.val_loss.reset_states()
                self.train_iou.reset_states()
                self.val_iou.reset_states()

                # Train Step
                train_loop = tqdm(total=self.train_dataset_size // self.batch_size, position=0, leave=True)
                for images, labels in self.train_dataset:
                    self.train_step(images, labels)
                    train_loop.set_description('Train - Epoch: {}, Loss: {:.4f}, IoU: {:.4f}'.format(
                        epoch, self.train_loss.result(), self.train_iou.result()))
                    train_loop.update(1)
                train_loop.close()

                # Validation Step
                val_loop = tqdm(total=self.val_dataset_size // self.batch_size, position=0, leave=True)
                for images, labels, in self.val_dataset:
                    self.val_step(images, labels)
                    val_loop.set_description('Val   - Epoch: {}, Loss: {:.4f}, IoU: {:.4f}'.format(
                        epoch, self.val_loss.result(), self.val_iou.result()))
                    val_loop.update(1)
                val_loop.close()

                train_losses.append(self.train_loss.result().numpy())
                val_losses.append(self.val_loss.result().numpy())
                train_ious.append(self.train_iou.result().numpy())
                val_ious.append(self.val_iou.result().numpy())

                # Only save the model if the validation IoU is greater than anything we've seen
                if val_ious[-1] > best_val_iou and epoch >= self.save_best_after - 1:
                    best_val_iou = val_ious[-1]
                    self.model.save_weights(self.save_path)
                    tf.print('\nSaving model to', self.save_path, '. Val:', best_val_iou)

        except Exception as e:
            print('Exception caught during training: {0}'.format(e))
        finally:
            return self.model, (train_losses, val_losses), (train_ious, val_ious)
