import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras import Model


class ConvBlock(Model):
    def __init__(self, filters, activation=kl.PReLU, dropout_rate=0.0, max_pool=True, name="ConvBlock"):
        super(ConvBlock, self).__init__()

        self.model = tf.keras.Sequential(name=name)
        self.model.add(kl.Conv2D(filters, kernel_size=(4, 4), padding='same'))
        self.model.add(kl.BatchNormalization(renorm=True))
        self.model.add(activation())

        if dropout_rate != 0.0:
            self.model.add(kl.Dropout(dropout_rate))

        if max_pool:
            self.model.add(kl.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    def call(self, x, **kwargs):
        return self.model(x, **kwargs)


class ResidualBlock(Model):
    def __init__(self, filters, activation=kl.PReLU):
        super(ResidualBlock, self).__init__()

        self.filters = filters
        self.act_final = activation()

        self.shortcut = kl.Conv2D(filters, kernel_size=(1, 1), use_bias=False)

        self.conv1 = kl.Conv2D(filters, kernel_size=(3, 3), padding='same')
        self.bn1 = kl.BatchNormalization(renorm=True)
        self.act1 = activation()

        self.conv2 = kl.Conv2D(filters, kernel_size=(3, 3), padding='same')
        self.bn2 = kl.BatchNormalization(renorm=True)
        self.act2 = activation()

        self.conv3 = kl.Conv2D(filters, kernel_size=(3, 3), padding='same')
        self.bn3 = kl.BatchNormalization(renorm=True)
        self.act3 = activation()

        self.conv4 = kl.Conv2D(filters, kernel_size=(3, 3), padding='same')

    def call(self, x, **kwargs):
        # Add shortcut if necessary
        if x.shape[-1] != self.filters:  # Channel Dimension
            x = self.shortcut(x)

        # Conv1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        # Conv2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        # Conv3
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)

        # Conv4 - Logits
        out = self.conv4(out)

        # Element-Wise Addition
        out = tf.math.add(out, x)

        # Final Activation
        out = self.act_final(out)

        return out


class ANet(Model):
    def __init__(self, activation=kl.PReLU, dropout_rate=0.5):
        super(ANet, self).__init__(name='A-Net')

        self.conv1 = ConvBlock(12, activation=activation, dropout_rate=dropout_rate, max_pool=True, name='conv1')
        self.conv2 = ConvBlock(16, activation=activation, dropout_rate=dropout_rate, max_pool=True, name='conv2')
        self.conv3 = ConvBlock(32, activation=activation, dropout_rate=dropout_rate, max_pool=True, name='conv3')
        self.conv4 = ConvBlock(2, activation=activation, dropout_rate=dropout_rate, max_pool=False, name='conv4')

    def call(self, x, **kwargs):
        out = self.conv1(x, **kwargs)
        out = self.conv2(out, **kwargs)
        out = self.conv3(out, **kwargs)
        out = self.conv4(out, **kwargs)

        out = tf.keras.activations.sigmoid(out)

        return out


class RUNet(Model):
    def __init__(self, initial_filters=8, activation=kl.PReLU):
        super(RUNet, self).__init__(name='RU-Net')

        self.block1 = ResidualBlock(filters=initial_filters, activation=activation)
        self.block2 = ResidualBlock(filters=initial_filters * 2, activation=activation)
        self.block3 = ResidualBlock(filters=initial_filters * 4, activation=activation)
        self.block4 = ResidualBlock(filters=initial_filters * 8, activation=activation)
        self.block5 = ResidualBlock(filters=initial_filters * 4, activation=activation)
        self.block6 = ResidualBlock(filters=initial_filters * 2, activation=activation)
        self.block7 = ResidualBlock(filters=initial_filters, activation=activation)

        self.conv_final = kl.Conv2D(filters=2, kernel_size=(1, 1), padding='same')

        self.mp1 = kl.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.mp2 = kl.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.mp3 = kl.MaxPooling2D(pool_size=(2, 2), padding='same')

        self.deconv1 = kl.Conv2DTranspose(initial_filters * 4, kernel_size=(2, 2), strides=(2, 2), padding='same')
        self.act1 = activation()
        self.deconv2 = kl.Conv2DTranspose(initial_filters * 2, kernel_size=(2, 2), strides=(2, 2), padding='same')
        self.act2 = activation()
        self.deconv3 = kl.Conv2DTranspose(initial_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')
        self.act3 = activation()

    def call(self, x, **kwargs):
        # Down
        block1_out = self.block1(x, **kwargs)
        block2_in = self.mp1(block1_out)

        block2_out = self.block2(block2_in, **kwargs)
        block3_in = self.mp2(block2_out)

        block3_out = self.block3(block3_in, **kwargs)
        block4_in = self.mp3(block3_out)

        # Bottom
        block4_out = self.block4(block4_in, **kwargs)

        # Up
        block5_in = self.deconv1(block4_out)
        block5_in = self.act1(block5_in)
        block5_out = self.block5(tf.concat((block5_in, block3_out), axis=3), **kwargs)

        block6_in = self.deconv2(block5_out)
        block6_in = self.act2(block6_in)
        block6_out = self.block6(tf.concat((block6_in, block2_out), axis=3), **kwargs)

        block7_in = self.deconv3(block6_out)
        block7_in = self.act3(block7_in)
        block7_out = self.block7(tf.concat((block7_in, block1_out), axis=3), **kwargs)

        # Final Conv to get down to 1 channel
        final_out = self.conv_final(block7_out)

        return final_out


class ARUNet(Model):
    def __init__(self):
        super(ARUNet, self).__init__()

        # Scale 1 (Normal Size)
        self.anet1 = ANet()
        self.runet1 = RUNet()

        # Scale 2
        self.mp1 = kl.MaxPooling2D(pool_size=(2, 2))
        self.deconv1 = kl.Conv2DTranspose(filters=1, kernel_size=(2, 2), strides=(2, 2), padding='same')
        self.anet2 = ANet()
        self.runet2 = RUNet()

        # Scale 3
        self.mp2 = kl.MaxPooling2D(pool_size=(2, 2))
        self.deconv2 = kl.Conv2DTranspose(filters=1, kernel_size=(2, 2), strides=(4, 4), padding='same')
        self.anet3 = ANet()
        self.runet3 = RUNet()

        # Scale 4
        self.mp3 = kl.MaxPooling2D(pool_size=(2, 2))
        self.deconv3 = kl.Conv2DTranspose(filters=1, kernel_size=(2, 2), strides=(8, 8), padding='same')
        self.anet4 = ANet()
        self.runet4 = RUNet()

        # Scale 5
        self.mp4 = kl.MaxPooling2D(pool_size=(2, 2))
        self.deconv4 = kl.Conv2DTranspose(filters=1, kernel_size=(2, 2), strides=(16, 16), padding='same')
        self.anet5 = ANet()
        self.runet5 = RUNet()

        self.softmax = kl.Softmax(axis=3)

    def call(self, x, **kwargs):
        # Scale 1
        anet1_out = self.anet1(x, **kwargs)
        runet1_out = self.runet1(x, **kwargs)
        arunet1_out = tf.math.multiply(anet1_out, runet1_out)

        # Scale 2
        x2 = self.mp1(x)
        anet2_out = self.anet2(x2, **kwargs)
        runet2_out = self.runet2(x2, **kwargs)
        arunet2_out = tf.math.multiply(anet2_out, runet2_out)
        arunet2_out = self.deconv1(arunet2_out)

        # Scale 3
        x3 = self.mp2(x2)
        anet3_out = self.anet3(x3, **kwargs)
        runet3_out = self.runet3(x3, **kwargs)
        arunet3_out = tf.math.multiply(anet3_out, runet3_out)
        arunet3_out = self.deconv2(arunet3_out)

        # Scale 4
        x4 = self.mp3(x3)
        anet4_out = self.anet4(x4, **kwargs)
        runet4_out = self.runet4(x4, **kwargs)
        arunet4_out = tf.math.multiply(anet4_out, runet4_out)
        arunet4_out = self.deconv3(arunet4_out)

        # Scale 5
        x5 = self.mp4(x4)
        anet5_out = self.anet5(x5, **kwargs)
        runet5_out = self.runet5(x5, **kwargs)
        arunet5_out = tf.math.multiply(anet5_out, runet5_out)
        arunet5_out = self.deconv4(arunet5_out)

        # Element-Wise Summation
        arunet_out = arunet1_out + arunet2_out + arunet3_out + arunet4_out + arunet5_out

        # Use sigmoid to give confidence level
        arunet_out = self.softmax(arunet_out)

        return arunet_out
