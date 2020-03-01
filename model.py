import tensorflow as tf
from tensorflow import nn
from tensorflow.contrib.layers import conv2d, conv2d_transpose, max_pool2d, batch_norm

def Unet(x):
    conv1 = conv2d(x, 32, 3, normalizer_fn=batch_norm)
    conv1 = conv2d(conv1, 32, 3, normalizer_fn=batch_norm)
    pool1 = max_pool2d(conv1, 2)

    conv2 = conv2d(pool1, 64, 3, normalizer_fn=batch_norm)
    conv2 = conv2d(conv2, 64, 3, normalizer_fn=batch_norm)
    pool2 = max_pool2d(conv2, 2)

    conv3 = conv2d(pool2, 128, 3, normalizer_fn=batch_norm)
    conv3 = conv2d(conv3, 128, 3, normalizer_fn=batch_norm)
    pool3 = max_pool2d(conv3, 2)

    conv4 = conv2d(pool3, 256, 3, normalizer_fn=batch_norm)
    conv4 = conv2d(conv4, 256, 3, normalizer_fn=batch_norm)
    pool4 = max_pool2d(conv4, 2)

    conv5 = conv2d(pool4, 512, 3, normalizer_fn=batch_norm)
    conv5 = conv2d(conv5, 512, 3, normalizer_fn=batch_norm)

    up6 = conv2d_transpose(conv5, 512, 3, stride=2, padding='SAME', normalizer_fn=batch_norm)
    up6 = tf.concat([up6, conv4], axis=-1)
    conv6 = conv2d(up6, 256, 3, normalizer_fn=batch_norm)
    conv6 = conv2d(conv6, 256, 3, normalizer_fn=batch_norm)

    up7 = conv2d_transpose(conv6, 256, 3, stride=2, padding='SAME', normalizer_fn=batch_norm)
    up7 = tf.concat([up7, conv3], axis=-1)
    conv7 = conv2d(up7, 128, 3, normalizer_fn=batch_norm)
    conv7 = conv2d(conv7, 128, 3, normalizer_fn=batch_norm)

    up8 = conv2d_transpose(conv7, 128, 3, stride=2, padding='SAME', normalizer_fn=batch_norm)
    up8 = tf.concat([up8, conv2], axis=-1)
    conv8 = conv2d(up8, 64, 3, normalizer_fn=batch_norm)
    conv8 = conv2d(conv8, 64, 3, normalizer_fn=batch_norm)

    up9 = conv2d_transpose(conv8, 64, 3, stride=2, padding='SAME', normalizer_fn=batch_norm)
    up9 = tf.concat([up9, conv1], axis=-1)
    conv9 = conv2d(up9, 32, 3, normalizer_fn=batch_norm)
    conv9 = conv2d(conv9, 32, 3, normalizer_fn=batch_norm)

    pred = conv2d(conv9, 2, 1)

    return pred
