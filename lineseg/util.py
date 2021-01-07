import tensorflow as tf


@tf.function
def model_inference(model, imgs):
    return model(imgs, training=False)
