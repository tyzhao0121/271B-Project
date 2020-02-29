import tensorflow as tf
from tensorflow.contrib.layers import conv2d, conv2d_transpose, max_pool2d, batch_norm

import os
import numpy as np
from PIL import Image

class AutoContext:

    def __init__(self, BATCH_SIZE, MAX_EPOCH):
        self.display_step = 2

        self.width = 256
        self.height = 256
        self.n_channels = 3
        self.n_classes = 2
        self.max_epoch = MAX_EPOCH

        self.batch_size = BATCH_SIZE
        self.learning_rate = 0.001
        self.loss_method = 'weighted_cross_entropy'
        self.modelPath = '/home/tiz007/271B-Project/ckpts'
        self.data_root = '/home/tiz007/lgg-mri-segmentation/kaggle_3m'

    def Unet(self, x):
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

    def generate_class_weight(self):
        """
        arg_labels = np.argmax(labels, axis = -1)
        class_weights = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            class_weights[i] = 1 / np.mean(arg_labels == i) ** 0.3
        class_weights /= np.sum(class_weights)
        """
        
        class_weights = np.ones(self.n_classes)

        return class_weights

    def generate_dataset(self, datapath):
        all_list = os.listdir(datapath)
        n = len(all_list)
        np.random.shuffle(all_list)
        f1 = open('train.txt', 'w')
        f2 = open('val.txt', 'w')
        f3 = open('test.txt', 'w')
        for i, name in enumerate(all_list):
            if (i <= n/3*2):
                print(name, file=f1)
            elif (i <= n/6*5):
                print(name, file=f2)
            else:
                print(name, file=f3)
        f1.close()
        f2.close()
        f3.close()

    def load_data(self):
        image_batch = np.zeros([self.batch_size, self.height, self.width, self.n_channels])
        label_batch = np.zeros([self.batch_size, self.height, self.width, self.n_classes])
        
        for _ in range(100):
            yield [image_batch, label_batch]

    def train(self):

        x = tf.placeholder(tf.float32, [None, self.width, self.height, self.n_channels])
        y = tf.placeholder(tf.float32, [None, self.width, self.height, self.n_classes])
        lr = tf.placeholder(tf.float32)
        weights = tf.placeholder(tf.float32, [self.batch_size * self.width * self.height])
        
        #define model:
        pred = self.Unet(x)

        # Define loss and optimizer
        pred_reshape = tf.reshape(pred, [self.batch_size * self.width * self.height, self.n_classes])
        y_reshape = tf.reshape(y, [self.batch_size * self.width * self.height, self.n_classes])

        if self.loss_method == 'cross_entropy':
            cost = tf.losses.softmax_cross_entropy(onehot_labels = y , logits = pred)

        elif self.loss_method == 'weighted_cross_entropy':
            cost = tf.losses.softmax_cross_entropy(onehot_labels = y_reshape , logits = pred_reshape, weights=weights)

        elif self.loss_method == 'dice':
            intersection = tf.reduce_sum(pred_reshape * y_reshape)
            cost = -(2 * intersection + 1)/(tf.reduce_sum(pred_reshape) + tf.reduce_sum(y_reshape) + 1)

        else:
            raise NotImplementedError

        optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred, -1), tf.argmax(y, -1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        class_weights = self.generate_class_weight()

        saver = tf.train.Saver()
        model_path = os.path.join(self.modelPath, 'Unet.ckpt')
        
        # Initializing the variables
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.max_epoch):
                for batch_idx, (image_batch, label_batch) in enumerate(self.load_data()): 
                    # flatten to n dimsion
                    label_vect = np.reshape(np.argmax(label_batch, axis=-1), [self.batch_size * self.width * self.height])
                    weight_vect = class_weights[label_vect]

                    # Fit training using batch data
                    feed_dict = {x: image_batch, y: label_batch, weights: weight_vect, lr:self.learning_rate}
                    loss, acc, _ = sess.run([cost, accuracy, optimizer], feed_dict=feed_dict)

                    if batch_idx % self.display_step == 0:
                        print("Epoch: %d, batch_idx: %d, Minibatch Loss: %0.6f , Training Accuracy: %0.5f " 
                            % (epoch, batch_idx, loss, acc))

                        # Save the variables to disk.
                        # saver.save(sess, model_path)
                    if epoch % 2 == 0:
                        self.learning_rate *= 0.9
