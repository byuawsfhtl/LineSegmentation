import tensorflow as tf
from tqdm import tqdm

from src.lineseg.model import ARUNet


class ModelTrainer:
    """
    ModelTrainer

    Responsible for training the model. Scope becomes an issues when dealing with @tf.function.
    It's easier to place all of the training code into an object so we don't run into issues.
    Once the object is created, the __call__ method will train and return the results and the
    trained model.
    """
    def __init__(self, epochs, batch_size, train_dataset, train_dataset_size, val_dataset, val_dataset_size, save_path,
                 lr=4e-4, weights_path=None, save_best_after=30, learning_rate_decay=.985):
        """
        Set up the necessary variables that will be used during training, including the model, optimizer,
        encoder, and other metrics.

        :param epochs:
        :param batch_size:
        :param train_dataset:
        :param train_dataset_size:
        :param val_dataset:
        :param val_dataset_size:
        :param lr:
        :param weights_path:
        :param save_best_after:
        :param learning_rate_decay:
        """

        self.epochs = epochs
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.train_dataset_size = train_dataset_size
        self.val_dataset = val_dataset
        self.val_dataset_size = val_dataset_size
        self.save_path = save_path
        self.save_best_after = save_best_after

        self.model = ARUNet()
        if weights_path is not None:
            self.model.load_weights(weights_path)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, 1, learning_rate_decay)
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr,)
        self.objective = tf.keras.losses.SparseCategoricalCrossentropy()

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
            predictions = self.model(images)
            loss = self.objective(labels, predictions)

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
        predictions = self.model(images)
        loss = self.objective(labels, predictions)

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
                train_loop = tqdm(total=self.train_dataset_size//self.batch_size, position=0, leave=True)
                for images, labels in self.train_dataset:
                    self.train_step(images, labels)
                    train_loop.set_description('Train - Epoch: {}, Loss: {:.4f}, IoU: {:.4f}'.format(epoch, self.train_loss.result(), self.train_iou.result()))
                    train_loop.update(1)
                train_loop.close()

                # Validation Step
                val_loop = tqdm(total=self.val_dataset_size//self.batch_size, position=0, leave=True)
                for images, labels, in self.val_dataset:
                    self.val_step(images, labels)
                    val_loop.set_description('Val   - Epoch: {}, Loss: {:.4f}, IoU: {:.4f}'.format(epoch, self.val_loss.result(), self.val_iou.result()))
                    val_loop.update(1)
                val_loop.close()

                train_losses.append(self.train_loss.result().numpy())
                val_losses.append(self.val_loss.result().numpy())
                train_ious.append(self.train_iou.result().numpy())
                val_ious.append(self.val_iou.result().numpy())

                # Only save the model if the validation IoU is greater than anything we've seen
                if val_ious[-1] > best_val_iou and epoch > self.save_best_after:
                    best_val_iou = val_ious[-1]
                    self.model.save(self.save_path)
                    tf.print('Saving model to', self.save_path, '. Val:', best_val_iou)

        except Exception as e:
            print('Exception caught during training: {0}'.format(e))
        finally:
            return self.model, (train_losses, val_losses), (train_ious, val_ious)
