import tensorflow as tf
from tensorflow import nn
from tensorflow.contrib.layers import conv2d, conv2d_transpose, max_pool2d, batch_norm

import os
import numpy as np
from PIL import Image


from model import Unet
from datafile import Dataset

class UnetModel:

    def __init__(self, BATCH_SIZE, MAX_EPOCH):
        self.display_step = 100

        self.width = 256
        self.height = 256
        self.n_channels = 3
        self.n_classes = 2
        self.max_epoch = MAX_EPOCH

        self.batch_size = BATCH_SIZE
        self.learning_rate = 0.001
        #'weighted_cross_entropy'
        self.loss_method = 'weighted_cross_entropy'

        self.modelPath = '../model'
        
        self.data = Dataset('../data/kaggle_3m', BATCH_SIZE, MAX_EPOCH)
       
        print("Data loading finished!\n")
        
        print("Max Epoch: {}, Batch Size: {}, Learning Rate: {}, Loss: {}".format(self.max_epoch, self.batch_size,
                                                                                  self.learning_rate, self.loss_method))

    def network(self, x):
        
        pred = Unet(x)
        
        return pred


    def train(self):
        
       
        x = tf.placeholder(tf.float32, [None, self.width, self.height, self.n_channels])
        y = tf.placeholder(tf.float32, [None, self.width, self.height, self.n_classes])
        lr = tf.placeholder(tf.float32)
        weights = tf.placeholder(tf.float32, [self.batch_size * self.width * self.height])
        
        #define model:
        pred = self.network(x)

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
        
        pred_lbl = tf.reshape(tf.argmin(pred, -1), [-1, self.width * self.height])
        y_lbl = tf.reshape(tf.argmin(y, -1), [-1, self.width * self.height])
        
        intersection = tf.reduce_sum(pred_lbl*y_lbl)
        sum_pred_lbl = tf.reduce_sum(pred_lbl)
        sum_y_lbl = tf.reduce_sum(y_lbl)
        diceco = (2 * intersection + 1)/(sum_pred_lbl + sum_y_lbl + 1)
        
        class_weights = self.data.get_class_weight()

        saver = tf.train.Saver()
        model_path = os.path.join(self.modelPath, 'Unet.ckpt')
        
        # Initializing the variables
        init = tf.global_variables_initializer()
        
        '''###########################
        print(np.max(train_img_list), np.min(train_img_list))
        print(train_img_list.shape)
        ###########################'''
        
        
        with tf.Session() as sess:
            sess.run(init)
            for batch_idx, (image_batch, label_batch) in enumerate(self.data.train_batch()): 
                # flatten to n dimsion
                label_vect = np.reshape(np.argmax(label_batch, axis=-1), [self.batch_size * self.width * self.height])
                weight_vect = class_weights[label_vect]

                # Fit training using batch data
                feed_dict = {x: image_batch, y: label_batch, weights: weight_vect, lr:self.learning_rate}
                dice, loss, acc, _, inter, sum_pred, sum_y = sess.run([diceco, cost, accuracy, optimizer, intersection, sum_pred_lbl, sum_y_lbl], feed_dict=feed_dict)

                
                if batch_idx % self.display_step == 0:
                    print("Batch_idx: %d, Minibatch Loss: %0.6f , Training Accuracy: %0.5f, Dice Coeffecient: %0.5f " 
                        % (batch_idx, loss, acc, dice))
                    print("intersection: {}, sum_pred: {}, sum_y: {}".format(inter, sum_pred, sum_y))
                    # Save the variables to disk.
                    # saver.save(sess, model_path)
                
                if batch_idx % 500 == 499:
                    #self.learning_rate *= 0.9
                    
                    num_val = 0
                    acc_sum = 0
                    dce_sum = 0
                    for val_batch_idx, (image_batch, label_batch) in enumerate(self.data.val_batch()): 
                   
                        feed_dict = {x: image_batch, y: label_batch, weights: weight_vect, lr:self.learning_rate}
                        dice, loss, acc = sess.run([diceco, cost, accuracy], feed_dict=feed_dict)
                        
                        num_val += image_batch.shape[0]
                        acc_sum += acc*image_batch.shape[0]
                        dce_sum += dice*image_batch.shape[0]
                        
                    val_acc = acc_sum/num_val
                    val_dce = dce_sum / num_val
                    print("validation acc: {}".format(val_acc))
                    print("validation dice: {}".format(val_dce))
                
                        