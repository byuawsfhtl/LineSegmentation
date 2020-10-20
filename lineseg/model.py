import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2

L2c = 0.0003  # The L2 Coefficient


class ConvActDropMp(Model):
    """
    Simple sequential layer that performs a convolution, activation, dropout, max pool.
    """
    def __init__(self, filters, activation=kl.ReLU, dropout_rate=0.3, max_pool=True, name="ConvBnActDropMp"):
        """
        :param filters: The number of filters used in the convolutional layers
        :param activation: The keras activation type
        :param dropout_rate: The rate of dropout
        :param max_pool: Boolean indicating whether or not max pooling is used at end
        :param name: The name of the operation
        """
        super(ConvActDropMp, self).__init__()

        self.model = tf.keras.Sequential(name=name)
        self.model.add(kl.Conv2D(filters, kernel_size=(4, 4), padding='same', kernel_regularizer=l2(L2c)))
        self.model.add(activation())

        if dropout_rate is not None and dropout_rate != 0.0:
            self.model.add(kl.Dropout(dropout_rate))

        if max_pool:
            self.model.add(kl.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    def call(self, x, **kwargs):
        return self.model(x, **kwargs)


class DeconvActDrop(Model):
    """
    Simple layer that performs deconvolution, activation, dropout
    """
    def __init__(self, filters, kernel_size=(2, 2), strides=(2, 2), activation=kl.ReLU, dropout_rate=0.3,
                 name="DeconvBnActDrop"):
        """
        :param filters: The number of filters used in convolutional layers
        :param kernel_size: The kernel size as tuple or int
        :param strides: The strides as tuple or int
        :param activation: The keras activation type
        :param dropout_rate: The rate of dropout
        :param name: The name of the operation
        """
        super(DeconvActDrop, self).__init__()

        self.model = tf.keras.Sequential(name=name)
        self.model.add(kl.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding='same',
                                          kernel_regularizer=l2(L2c)))
        self.model.add(activation())

        if dropout_rate is not None and dropout_rate != 0.0:
            self.model.add(kl.Dropout(dropout_rate))

    def call(self, x, **kwargs):
        return self.model(x, **kwargs)


class ResidualBlock(Model):
    """
    Residual Block layer used in ARU-Net
    """
    def __init__(self, filters, activation=kl.ReLU, dropout_rate=0.3):
        """
        :param filters: The number of filters used in convolutions
        :param activation: The keras activation type
        :param dropout_rate: The rate of dropout
        """
        super(ResidualBlock, self).__init__()

        self.filters = filters
        self.act_final = activation()

        self.shortcut = kl.Conv2D(filters, kernel_size=(1, 1), use_bias=False)

        self.conv = tf.keras.Sequential()
        self.conv.add(kl.Conv2D(filters, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(L2c)))
        self.conv.add(activation())
        if dropout_rate is not None and dropout_rate != 0.0:
            self.conv.add(kl.Dropout(dropout_rate))

        self.conv.add(kl.Conv2D(filters, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(L2c)))
        self.conv.add(activation())
        if dropout_rate is not None and dropout_rate != 0.0:
            self.conv.add(kl.Dropout(dropout_rate))

        self.conv.add(kl.Conv2D(filters, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(L2c)))
        self.conv.add(activation())
        if dropout_rate is not None and dropout_rate != 0.0:
            self.conv.add(kl.Dropout(dropout_rate))

        self.conv4 = kl.Conv2D(filters, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(L2c))

    def call(self, x, **kwargs):
        # Add shortcut if necessary
        if x.shape[-1] != self.filters:  # Channel Dimension
            x = self.shortcut(x)

        # Send through the conv block
        out = self.conv(x, **kwargs)

        # Element-Wise Addition
        out = tf.math.add(out, x)

        # Final Activation
        out = self.act_final(out)

        return out


class ANet(Model):
    """
    A-Net model used in ARU-Net
    """
    def __init__(self, activation=kl.ReLU, dropout_rate=0.3):
        """
        :param activation: The keras activation type
        :param dropout_rate: The rate of dropout
        """
        super(ANet, self).__init__(name='A-Net')

        self.conv1 = ConvActDropMp(12, activation=activation, dropout_rate=dropout_rate, max_pool=True, name='conv1')
        self.conv2 = ConvActDropMp(16, activation=activation, dropout_rate=dropout_rate, max_pool=True, name='conv2')
        self.conv3 = ConvActDropMp(32, activation=activation, dropout_rate=dropout_rate, max_pool=True, name='conv3')
        self.conv4 = ConvActDropMp(2, activation=activation, dropout_rate=dropout_rate, max_pool=False, name='conv4')

    def call(self, x, **kwargs):
        out = self.conv1(x, **kwargs)
        out = self.conv2(out, **kwargs)
        out = self.conv3(out, **kwargs)
        out = self.conv4(out, **kwargs)

        return out


class RUNet(Model):
    """
    RU-Net Model used in ARU-Net.
    """
    def __init__(self, initial_filters=8, activation=kl.ReLU, dropout_rate=0.3):
        """
        :param initial_filters: The initial number of convolutional filters used in residual blocks
        :param activation: The activation type
        :param dropout_rate: The rate of dropout
        """
        super(RUNet, self).__init__(name='RU-Net')

        self.block1 = ResidualBlock(filters=initial_filters, activation=activation, dropout_rate=dropout_rate)
        self.block2 = ResidualBlock(filters=initial_filters * 2, activation=activation, dropout_rate=dropout_rate)
        self.block3 = ResidualBlock(filters=initial_filters * 4, activation=activation, dropout_rate=dropout_rate)
        self.block4 = ResidualBlock(filters=initial_filters * 8, activation=activation, dropout_rate=dropout_rate)
        self.block5 = ResidualBlock(filters=initial_filters * 16, activation=activation, dropout_rate=dropout_rate)
        self.block6 = ResidualBlock(filters=initial_filters * 32, activation=activation, dropout_rate=dropout_rate)
        self.block7 = ResidualBlock(filters=initial_filters * 16, activation=activation, dropout_rate=dropout_rate)
        self.block8 = ResidualBlock(filters=initial_filters * 8, activation=activation, dropout_rate=dropout_rate)
        self.block9 = ResidualBlock(filters=initial_filters * 4, activation=activation, dropout_rate=dropout_rate)
        self.block10 = ResidualBlock(filters=initial_filters * 2, activation=activation, dropout_rate=dropout_rate)
        self.block11 = ResidualBlock(filters=initial_filters, activation=activation, dropout_rate=dropout_rate)

        self.conv_final = kl.Conv2D(filters=2, kernel_size=(1, 1), padding='same')

        self.mp1 = kl.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.mp2 = kl.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.mp3 = kl.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.mp4 = kl.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.mp5 = kl.MaxPooling2D(pool_size=(2, 2), padding='same')

        self.deconv1 = DeconvActDrop(initial_filters * 16, kernel_size=(3, 3), strides=(2, 2), activation=activation,
                                     dropout_rate=dropout_rate)
        self.deconv2 = DeconvActDrop(initial_filters * 8, kernel_size=(3, 3), strides=(2, 2), activation=activation,
                                     dropout_rate=dropout_rate)
        self.deconv3 = DeconvActDrop(initial_filters * 4, kernel_size=(3, 3), strides=(2, 2), activation=activation,
                                     dropout_rate=dropout_rate)
        self.deconv4 = DeconvActDrop(initial_filters * 2, kernel_size=(3, 3), strides=(2, 2), activation=activation,
                                     dropout_rate=dropout_rate)
        self.deconv5 = DeconvActDrop(initial_filters * 1, kernel_size=(3, 3), strides=(2, 2), activation=activation,
                                     dropout_rate=dropout_rate)

    def call(self, x, **kwargs):
        # Down
        block1_out = self.block1(x, **kwargs)
        block2_in = self.mp1(block1_out)

        block2_out = self.block2(block2_in, **kwargs)
        block3_in = self.mp2(block2_out)

        block3_out = self.block3(block3_in, **kwargs)
        block4_in = self.mp3(block3_out)

        block4_out = self.block4(block4_in, **kwargs)
        block5_in = self.mp4(block4_out)

        block5_out = self.block5(block5_in, **kwargs)
        block6_in = self.mp5(block5_out)

        # Bottom
        block6_out = self.block6(block6_in, **kwargs)

        # Up
        block7_in = self.deconv1(block6_out, **kwargs)
        block7_out = self.block7(tf.concat((block7_in, block5_out), axis=3), **kwargs)

        block8_in = self.deconv2(block7_out, **kwargs)
        block8_out = self.block8(tf.concat((block8_in, block4_out), axis=3), **kwargs)

        block9_in = self.deconv3(block8_out, **kwargs)
        block9_out = self.block9(tf.concat((block9_in, block3_out), axis=3), **kwargs)

        block10_in = self.deconv4(block9_out, **kwargs)
        block10_out = self.block10(tf.concat((block10_in, block2_out), axis=3), **kwargs)

        block11_in = self.deconv5(block10_out, **kwargs)
        block11_out = self.block11(tf.concat((block11_in, block1_out), axis=3), **kwargs)

        # Final Conv to get down to 1 channel
        final_out = self.conv_final(block11_out)

        return final_out


class ARUNet(Model):
    """
    ARU-Net model used for segmentation
    """
    def __init__(self, activation=kl.ReLU, runet_initial_filters=8, dropout_rate=0.0):
        """
        :param activation: The keras activation type
        :param runet_initial_filters: The initial number of convolutional filters used in residual blocks
        :param dropout_rate: The rate of dropout
        """
        super(ARUNet, self).__init__()

        # Scale 1 (Normal Size)
        self.anet = ANet(activation=activation, dropout_rate=dropout_rate)
        self.runet = RUNet(activation=activation, initial_filters=runet_initial_filters, dropout_rate=dropout_rate)

        # Scale 2
        self.ap1 = kl.AveragePooling2D(pool_size=(2, 2))
        self.a_deconv1 = DeconvActDrop(1, kernel_size=(2, 2), strides=(2, 2), dropout_rate=dropout_rate)
        self.ru_deconv1 = DeconvActDrop(1, kernel_size=(2, 2), strides=(2, 2), dropout_rate=dropout_rate)

        # Scale 3
        self.ap2 = kl.AveragePooling2D(pool_size=(2, 2))
        self.a_deconv2 = DeconvActDrop(1, kernel_size=(2, 2), strides=(4, 4), dropout_rate=dropout_rate)
        self.ru_deconv2 = DeconvActDrop(1, kernel_size=(2, 2), strides=(4, 4), dropout_rate=dropout_rate)

        # Scale 4
        self.ap3 = kl.AveragePooling2D(pool_size=(2, 2))
        self.a_deconv3 = DeconvActDrop(1, kernel_size=(2, 2), strides=(8, 8), dropout_rate=dropout_rate)
        self.ru_deconv3 = DeconvActDrop(1, kernel_size=(2, 2), strides=(8, 8), dropout_rate=dropout_rate)

        self.softmax = kl.Softmax(axis=3)

    def call(self, x, **kwargs):
        # Scale 1
        anet1_out = self.anet(x, **kwargs)
        runet1_out = self.runet(x, **kwargs)
        anet1_out = self.softmax(anet1_out)

        arunet1_out = tf.math.multiply(anet1_out, runet1_out)

        # Scale 2
        x2 = self.ap1(x)  # Rescale with average pooling

        anet2_out = self.anet(x2, **kwargs)  # Send through Attention Net
        runet2_out = self.runet(x2, **kwargs)  # Send through Residual U-Net

        anet2_out = self.a_deconv1(anet2_out, **kwargs)  # Deconvolve to normal image size
        runet2_out = self.ru_deconv1(runet2_out, **kwargs)
        anet2_out = self.softmax(anet2_out)  # Apply softmax to A-Net

        arunet2_out = tf.math.multiply(anet2_out, runet2_out)  # Combine A/RU

        # Scale 3
        x3 = self.ap2(x2)

        anet3_out = self.anet(x3, **kwargs)
        runet3_out = self.runet(x3, **kwargs)

        anet3_out = self.a_deconv2(anet3_out, **kwargs)
        runet3_out = self.ru_deconv2(runet3_out, **kwargs)
        anet3_out = self.softmax(anet3_out)

        arunet3_out = tf.math.multiply(anet3_out, runet3_out)

        # Scale 4
        x4 = self.ap3(x3)

        anet4_out = self.anet(x4, **kwargs)
        runet4_out = self.runet(x4, **kwargs)

        anet4_out = self.a_deconv3(anet4_out, **kwargs)
        runet4_out = self.ru_deconv3(runet4_out, **kwargs)
        anet4_out = self.softmax(anet4_out)

        arunet4_out = tf.math.multiply(anet4_out, runet4_out)

        # Element-Wise Summation
        arunet_out = arunet1_out + arunet2_out + arunet3_out + arunet4_out

        # Use softmax to give confidence level
        arunet_out = self.softmax(arunet_out)

        return arunet_out
