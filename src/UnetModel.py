import tensorflow as tf
from tensorflow import nn
from tensorflow.contrib.layers import conv2d, conv2d_transpose, max_pool2d, batch_norm

import os
import time

import numpy as np
from PIL import Image


from model import Unet
from datafile import Dataset

class UnetModel:

    def __init__(self, BATCH_SIZE, MAX_EPOCH):
        # display setting
        self.display_step = 100

        # image configuration
        self.width = 256
        self.height = 256
        self.n_channels = 3
        self.n_classes = 2
        
        # hyper parameters
        self.max_epoch = MAX_EPOCH
        self.batch_size = BATCH_SIZE
        self.learning_rate = 0.001
        self.loss_method = 'weighted_cross_entropy'

        self.model_name = 'Unet'
        
        self.logPath = '../log'
                
        self.modelPath = '../model'

        self.log_file = ''
        
        self.modelDir = ''
                
        self.data = Dataset('../data/kaggle_3m', BATCH_SIZE, MAX_EPOCH)
    
        print("Data loading finished!\n")
        
    def initial_log_and_modeldir(self):
        
        index = 0
        
        while self.model_name + '_log_' + str(index) + '.txt' in os.listdir(self.logPath):
            index += 1
        file_name = self.model_name + '_log_' + str(index) + '.txt'

        self.modelDir = os.path.join(self.modelPath, self.model_name + '_model_' + str(index))

        if not os.path.exists(self.modelDir):
            os.mkdir(self.modelDir)

        self.log_file = os.path.join(self.logPath, file_name)
            
        self.log(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        self.log("Max Epoch: {}, Batch Size: {}, Learning Rate: {}, Loss: {}".format(self.max_epoch, self.batch_size,
                                                                                  self.learning_rate, self.loss_method))
                             
    def network(self, x):
        
        pred = Unet(x)
        
        return pred

    def log(self, str):
        
        with open(self.log_file, 'a') as f:
            f.write(str + '\n')
        print(str)
    
    def train(self):
        self.initial_log_and_modeldir()
        x = tf.placeholder(tf.float32, [None, self.width, self.height, self.n_channels])
        y = tf.placeholder(tf.float32, [None, self.width, self.height, self.n_classes])
        lr = tf.placeholder(tf.float32)
        weights = tf.placeholder(tf.float32, [self.batch_size * self.width * self.height])
        
        with tf.variable_scope('Stage1'):
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
        
        # initial training log
        num_train = 0
        acc_sum_train = 0
        dce_sum_train = 0
        loss_sum_train = 0
        
        with tf.Session() as sess:
            sess.run(init)
            max_val_dce = 0.7
            for batch_idx, (image_batch, label_batch) in enumerate(self.data.train_batch()): 
                # flatten to n dimsion
                label_vect = np.reshape(np.argmax(label_batch, axis=-1), [self.batch_size * self.width * self.height])
                weight_vect = class_weights[label_vect]

                # Fit training using batch data
                feed_dict = {x: image_batch, y: label_batch, weights: weight_vect, lr:self.learning_rate}
                
                # run session
                train_pred, dice, loss, acc, _, inter, sum_pred, sum_y = sess.run([pred, diceco, cost, accuracy, optimizer, intersection, sum_pred_lbl, sum_y_lbl], feed_dict=feed_dict)

                # accumulate training result
                n_sample = image_batch.shape[0]
                num_train += n_sample
                acc_sum_train += acc * n_sample
                dce_sum_train += dice * n_sample
                loss_sum_train += loss * n_sample
                
                if batch_idx % self.display_step == 0:
                    train_acc = acc_sum_train / num_train
                    train_dce = dce_sum_train / num_train
                    train_loss = loss_sum_train / num_train
                    
                    self.log("Batch_idx: %d, Minibatch Loss: %0.6f, Training Accuracy: %0.5f, Dice Coeffecient: %0.5f " 
                        % (batch_idx, train_loss, train_acc, train_dce))
                    
                    num_train = 0
                    acc_sum_train = 0
                    dce_sum_train = 0
                    loss_sum_train = 0

                
                if batch_idx % 500 == 499 and train_dce > 0.5:
                    
                    
                    num_val = 0
                    acc_sum_val = 0
                    dce_sum_val = 0
                    for val_batch_idx, (image_batch, label_batch) in enumerate(self.data.val_batch()): 
                   
                        feed_dict = {x: image_batch, y: label_batch, weights: weight_vect, lr:self.learning_rate}
                        val_pred, dice, loss, acc = sess.run([pred, diceco, cost, accuracy], feed_dict=feed_dict)
                        
                        n_sample = image_batch.shape[0]
                        num_val += n_sample
                        acc_sum_val += acc * n_sample
                        dce_sum_val += dice * n_sample
                        
                    val_acc = acc_sum_val / num_val
                    val_dce = dce_sum_val / num_val
                    self.log("validation acc: {}".format(val_acc))
                    self.log("validation dice: {}".format(val_dce))
                    
                    # Save the variables to disk.
                    if val_dce > max_val_dce:
                        max_val_dce = val_dce
                        checkpoint_name = "{}_step_{}_dice_{:.4f}".format(self.model_name, batch_idx+1, val_dce)
                        checkpoint_path = os.path.join(self.modelDir, checkpoint_name)
                        saver.save(sess, checkpoint_path)
                        np.save("{}_step_{}_dice_{:.4f}_train_pred".format(self.model_name, batch_idx+1, val_dce), train_pred)
                        np.save("{}_step_{}_dice_{:.4f}_val_pred".format(self.model_name, batch_idx+1, val_dce), val_pred)
                        
                        self.log("saving checkpoint to {}".format(checkpoint_path))
                        
                if batch_idx % 5000 == 4999:
                    self.learning_rate *= 0.9

    def test(self, index, steps):
        print("start testing")
        model_dir = os.path.join(self.modelPath, "{}_model_{}".format(self.model_name, index)) #, "{}_model_step_{}".format(self.model_name, steps
        suffix = "{}_step_{}_dice".format(self.model_name, steps)
        
        meta_path = ''
        ckpt_path = ''
        for ckpt_file in os.listdir(model_dir):
            if ckpt_file[:len(suffix)] == suffix and ckpt_file[-4:] == 'meta':
                meta_path = os.path.join(model_dir, ckpt_file)
                ckpt_path = os.path.join(model_dir, ckpt_file[:-5])
                
        if meta_path == '' or ckpt_path == '':
            raise Exception("checkpoint not found")
        
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess, ckpt_path)
            graph = tf.get_default_graph()
            #for name in [n.name for n in tf.get_default_graph().as_graph_def().node]:
                #print(name)
            x = graph.get_tensor_by_name('Placeholder:0')
            y = graph.get_tensor_by_name('Placeholder_1:0')
            diceco = graph.get_tensor_by_name('truediv:0')
            accuracy = graph.get_tensor_by_name('Mean:0')
            for i in tf.get_collection(tf.GraphKeys.SUMMARIES):
                print(i)   # i.name if you want just a name
            num_val = 0
            acc_sum_val = 0
            dce_sum_val = 0
            for val_batch_idx, (image_batch, label_batch) in enumerate(self.data.test_batch()): 

                feed_dict = {x: image_batch, y: label_batch}
                dice, acc = sess.run([diceco, accuracy], feed_dict=feed_dict)

                n_sample = image_batch.shape[0]
                num_val += n_sample
                acc_sum_val += acc * n_sample
                dce_sum_val += dice * n_sample

            val_acc = acc_sum_val / num_val
            val_dce = dce_sum_val / num_val
            print("validation acc: {}".format(val_acc))
            print("validation dice: {}".format(val_dce))
        
        
                        
                
                        