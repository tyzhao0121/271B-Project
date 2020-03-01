import tensorflow as tf
from tensorflow import nn
from tensorflow.contrib.layers import conv2d, conv2d_transpose, max_pool2d, batch_norm

import os
import time

import numpy as np
from PIL import Image


from model import Unet
from datafile import Dataset

class AutoUnet:

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

        self.modelPath = '../model'
        self.model_name = 'AutoUnet'
        self.logPath = '../log'
        self.modelDir = None
        self.log_file = None
        self.initial_log()

        
        if not os.path.exists(self.modelDir):
            os.mkdir(self.modelDir)

        self.data = Dataset('../data/kaggle_3m', BATCH_SIZE, MAX_EPOCH)
        print("Data loading finished!\n")
        

        self.train_stage1_mask = np.ones((self.batch_size, self.width, self.height, 2)) / 2
        self.train_stage2_mask = np.zeros((self.batch_size, self.width, self.height, 2))
        self.train_stage3_mask = np.zeros((self.batch_size, self.width, self.height, 2))
        self.train_stage4_mask = np.zeros((self.batch_size, self.width, self.height, 2))
        self.val_stage1_mask = np.ones((self.batch_size, self.width, self.height, 2)) / 2
        self.val_stage2_mask = np.zeros((self.batch_size, self.width, self.height, 2))
        self.val_stage3_mask = np.zeros((self.batch_size, self.width, self.height, 2))
        self.val_stage4_mask = np.zeros((self.batch_size, self.width, self.height, 2))
    def get_one_batch(self):
        for (img, lbl) in self.data.train_batch():
            yield img, lbl
    def initial_log(self):
        
        i = 0
        while self.model_name + '_log_' + str(i) + '.txt' in os.listdir(self.logPath):
            i += 1
        file_name = self.model_name + '_log_' + str(i) + '.txt'
        
        
        self.modelDir = os.path.join(self.modelPath, self.model_name + '_model_' + str(i))
        self.log_file = os.path.join(self.logPath, file_name)
        return os.path.join(self.logPath, file_name)
                     
    def network(self, x):
        
        pred = Unet(x)
        
        return pred

    def log(self, str):
        
        with open(self.log_file, 'a') as f:
            f.write(str + '\n')
        print(str)
    def train_stage1(self):
        self.learning_rate = 0.001
        print(self.initial_log())
        self.log(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        self.log("Stage 1, Max Epoch: {}, Batch Size: {}, Learning Rate: {}, Loss: {}".format(self.max_epoch, self.batch_size, self.learning_rate, self.loss_method))
        
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, [None, self.width, self.height, self.n_channels+2])
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


        '''all_vars = tf.global_variables()
        print(all_vars)'''
        
            
        with tf.Session() as sess:
            sess.run(init)
            max_val_dce = 0.5
            for batch_idx, (image_batch, label_batch) in enumerate(self.data.train_batch()): 
                # flatten to n dimsion
                
                concated_image = np.concatenate((image_batch, self.train_stage1_mask), axis=3)
                
                label_vect = np.reshape(np.argmax(label_batch, axis=-1), [self.batch_size * self.width * self.height])
                weight_vect = class_weights[label_vect]

                # Fit training using batch data
                feed_dict = {x: concated_image, y: label_batch, weights: weight_vect, lr:self.learning_rate}

                # run session
                train_pred_mask, dice, loss, acc, _, inter, sum_pred, sum_y = sess.run([pred, diceco, cost, accuracy, optimizer, intersection, sum_pred_lbl, sum_y_lbl], feed_dict=feed_dict)

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

                    self.log("【Stage1】 Batch_idx: %d, Minibatch Loss: %0.6f, Training Accuracy: %0.5f, Dice Coeffecient: %0.5f " 
                        % (batch_idx, train_loss, train_acc, train_dce))

                    num_train = 0
                    acc_sum_train = 0
                    dce_sum_train = 0
                    loss_sum_train = 0

                if batch_idx % 5000 == 4999:
                    self.learning_rate *= 0.9
                    
                if batch_idx % 500 == 499:
                    
                    num_val = 0
                    acc_sum_val = 0
                    dce_sum_val = 0
                    for val_batch_idx, (image_batch, label_batch) in enumerate(self.data.val_batch()): 

                        concated_image = np.concatenate((image_batch, self.val_stage1_mask), axis=3)                      
                        
                        feed_dict = {x: concated_image, y: label_batch, weights: weight_vect, lr:self.learning_rate}
                        val_pred_mask, dice, loss, acc = sess.run([pred, diceco, cost, accuracy], feed_dict=feed_dict)

                        n_sample = image_batch.shape[0]
                        num_val += n_sample
                        acc_sum_val += acc * n_sample
                        dce_sum_val += dice * n_sample

                    val_acc = acc_sum_val / num_val
                    val_dce = dce_sum_val / num_val
                    self.log("【Stage1】validation acc: {}".format(val_acc))
                    self.log("【Stage1】validation dice: {}".format(val_dce))

                    # Save the variables to disk.
                    if val_dce > max_val_dce:
                        checkpoint_name = "{}_step_{}_dice_{:.4f}".format(self.model_name+'stage1', batch_idx+1, val_dce)
                        
                        self.train_stage2_mask = train_pred_mask
                        self.val_stage2_mask = val_pred_mask
                        
                        checkpoint_path = os.path.join(self.modelDir, checkpoint_name)
                        saver.save(sess, checkpoint_path, write_meta_graph=False)

                        self.log("【Stage1】saving checkpoint to {}".format(checkpoint_path))
                        
                        max_val_dce = val_dce
                        
                    if val_dce > 0.75:
                        return

    def train_stage2(self):
        self.learning_rate = 0.001
        print(self.initial_log())
        self.log(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        self.log("Stage 1, Max Epoch: {}, Batch Size: {}, Learning Rate: {}, Loss: {}".format(self.max_epoch, self.batch_size, self.learning_rate, self.loss_method))
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, [None, self.width, self.height, self.n_channels+2])
        y = tf.placeholder(tf.float32, [None, self.width, self.height, self.n_classes])
        lr = tf.placeholder(tf.float32)
        weights = tf.placeholder(tf.float32, [self.batch_size * self.width * self.height])

        with tf.variable_scope('Stage2'):
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


        '''all_vars = tf.global_variables()
        print(all_vars)'''
        
            
        with tf.Session() as sess:
            sess.run(init)
            max_val_dce = 0.5
            for batch_idx, (image_batch, label_batch) in enumerate(self.data.train_batch()): 
                # flatten to n dimsion
                
                concated_image = np.concatenate((image_batch, self.train_stage2_mask), axis=3)
                
                label_vect = np.reshape(np.argmax(label_batch, axis=-1), [self.batch_size * self.width * self.height])
                weight_vect = class_weights[label_vect]

                # Fit training using batch data
                feed_dict = {x: concated_image, y: label_batch, weights: weight_vect, lr:self.learning_rate}

                # run session
                train_pred_mask, dice, loss, acc, _, inter, sum_pred, sum_y = sess.run([pred, diceco, cost, accuracy, optimizer, intersection, sum_pred_lbl, sum_y_lbl], feed_dict=feed_dict)

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

                    self.log("【Stage2】Batch_idx: %d, Minibatch Loss: %0.6f, Training Accuracy: %0.5f, Dice Coeffecient: %0.5f " 
                        % (batch_idx, train_loss, train_acc, train_dce))

                    num_train = 0
                    acc_sum_train = 0
                    dce_sum_train = 0
                    loss_sum_train = 0

                    
                if batch_idx % 5000 == 4999:
                    self.learning_rate *= 0.9

                if batch_idx % 500 == 499:
                    

                    num_val = 0
                    acc_sum_val = 0
                    dce_sum_val = 0
                    for val_batch_idx, (image_batch, label_batch) in enumerate(self.data.val_batch()): 

                        concated_image = np.concatenate((image_batch, self.val_stage2_mask), axis=3)                      
                        
                        feed_dict = {x: concated_image, y: label_batch, weights: weight_vect, lr:self.learning_rate}
                        val_pred_mask, dice, loss, acc = sess.run([pred, diceco, cost, accuracy], feed_dict=feed_dict)

                        n_sample = image_batch.shape[0]
                        num_val += n_sample
                        acc_sum_val += acc * n_sample
                        dce_sum_val += dice * n_sample

                    val_acc = acc_sum_val / num_val
                    val_dce = dce_sum_val / num_val
                    self.log("【Stage2】validation acc: {}".format(val_acc))
                    self.log("【Stage2】validation dice: {}".format(val_dce))

                    # Save the variables to disk.
                    if val_dce > max_val_dce:
                        checkpoint_name = "{}_step_{}_dice_{:.4f}".format(self.model_name+'stage2', batch_idx+1, val_dce)
                        
                        self.train_stage3_mask = train_pred_mask
                        self.val_stage3_mask = val_pred_mask
                        
                        checkpoint_path = os.path.join(self.modelDir, checkpoint_name)
                        saver.save(sess, checkpoint_path, write_meta_graph=False)

                        self.log("【Stage2】saving checkpoint to {}".format(checkpoint_path))
                        
                        max_val_dce = val_dce
                        
                    if val_dce > 0.75:
                        return

                    
    def train_stage3(self):
        self.learning_rate = 0.001
        print(self.initial_log())
        self.log(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        self.log("Stage 1, Max Epoch: {}, Batch Size: {}, Learning Rate: {}, Loss: {}".format(self.max_epoch, self.batch_size, self.learning_rate, self.loss_method))
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, [None, self.width, self.height, self.n_channels+2])
        y = tf.placeholder(tf.float32, [None, self.width, self.height, self.n_classes])
        lr = tf.placeholder(tf.float32)
        weights = tf.placeholder(tf.float32, [self.batch_size * self.width * self.height])

        with tf.variable_scope('Stage3'):
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


        '''all_vars = tf.global_variables()
        print(all_vars)'''
        
            
        with tf.Session() as sess:
            sess.run(init)
            max_val_dce = 0.5
            for batch_idx, (image_batch, label_batch) in enumerate(self.data.train_batch()): 
                # flatten to n dimsion
                concated_image = np.concatenate((image_batch, self.train_stage3_mask), axis=3)
                
               
                label_vect = np.reshape(np.argmax(label_batch, axis=-1), [self.batch_size * self.width * self.height])
                weight_vect = class_weights[label_vect]

                # Fit training using batch data
                feed_dict = {x: concated_image, y: label_batch, weights: weight_vect, lr:self.learning_rate}

                # run session
                train_pred_mask, dice, loss, acc, _, inter, sum_pred, sum_y = sess.run([pred, diceco, cost, accuracy, optimizer, intersection, sum_pred_lbl, sum_y_lbl], feed_dict=feed_dict)

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

                    self.log("【Stage3】Batch_idx: %d, Minibatch Loss: %0.6f, Training Accuracy: %0.5f, Dice Coeffecient: %0.5f " 
                        % (batch_idx, train_loss, train_acc, train_dce))

                    num_train = 0
                    acc_sum_train = 0
                    dce_sum_train = 0
                    loss_sum_train = 0

                if batch_idx % 5000 == 4999:
                    self.learning_rate *= 0.9

                if batch_idx % 500 == 499:
   

                    num_val = 0
                    acc_sum_val = 0
                    dce_sum_val = 0
                    for val_batch_idx, (image_batch, label_batch) in enumerate(self.data.val_batch()): 
                    
                        concated_image = np.concatenate((image_batch, self.val_stage3_mask), axis=3)                      
                        
                        feed_dict = {x: concated_image, y: label_batch, weights: weight_vect, lr:self.learning_rate}
                        val_pred_mask, dice, loss, acc = sess.run([pred, diceco, cost, accuracy], feed_dict=feed_dict)
                        


                        n_sample = image_batch.shape[0]
                        num_val += n_sample
                        acc_sum_val += acc * n_sample
                        dce_sum_val += dice * n_sample

                    val_acc = acc_sum_val / num_val
                    val_dce = dce_sum_val / num_val
                    self.log("【Stage3】validation acc: {}".format(val_acc))
                    self.log("【Stage3】validation dice: {}".format(val_dce))

                    # Save the variables to disk.
                    if val_dce > max_val_dce:
                        checkpoint_name = "{}_step_{}_dice_{:.4f}".format(self.model_name+'stage3', batch_idx+1, val_dce)
                        
                        self.train_stage4_mask = train_pred_mask
                        self.val_stage4_mask = val_pred_mask
                        
                        checkpoint_path = os.path.join(self.modelDir, checkpoint_name)
                        saver.save(sess, checkpoint_path, write_meta_graph=False)

                        self.log("【Stage3】saving checkpoint to {}".format(checkpoint_path))
                        
                        max_val_dce = val_dce
                        
                    if val_dce > 0.75:
                        return
                    
    def train_end_to_end(self, file1, file2, file3):
        print(self.initial_log())
        self.log(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        self.log("Stage ALL, Max Epoch: {}, Batch Size: {}, Learning Rate: {}, Loss: {}".format(self.max_epoch, self.batch_size, self.learning_rate, self.loss_method))
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, [None, self.width, self.height, self.n_channels])
        y = tf.placeholder(tf.float32, [None, self.width, self.height, self.n_classes])
        lr = tf.placeholder(tf.float32)
        weights = tf.placeholder(tf.float32, [self.batch_size * self.width * self.height])

        with tf.variable_scope('Stage1'):
            #define model:
            x1 = tf.concat([x, tf.ones((self.batch_size, self.width, self.height, 2))/2], -1)
            print(x1)
            pred1 = self.network(x1)
        
        with tf.variable_scope('Stage2'):
            #define model:
            x2 = tf.concat([x, pred1], -1)
            print(x2)
            pred2 = self.network(x2)
        with tf.variable_scope('Stage3'):
            #define model:
            x3 = tf.concat([x, pred2], -1)
            print(x3)
            pred = self.network(x3)                                        
                        
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


        '''all_vars = tf.global_variables()
        print(all_vars)'''
        
        stage1_parameter = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Stage1')
        stage2_parameter = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Stage2')
        stage3_parameter = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Stage3')        
        
        saver1 = tf.train.Saver(stage1_parameter)
        saver2 = tf.train.Saver(stage2_parameter)
        saver3 = tf.train.Saver(stage3_parameter)
        
        
        
        with tf.Session() as sess:
            self.learning_rate = 1e-5
            sess.run(init)
            saver1.restore(sess, file1)
            saver2.restore(sess, file2)
            saver3.restore(sess, file3)
            
            max_val_dce = 0.5
            for batch_idx, (image_batch, label_batch) in enumerate(self.data.train_batch()): 
                # flatten to n dimsion
                
                
                label_vect = np.reshape(np.argmax(label_batch, axis=-1), [self.batch_size * self.width * self.height])
                weight_vect = class_weights[label_vect]

                # Fit training using batch data
                feed_dict = {x: image_batch, y: label_batch, weights: weight_vect, lr:self.learning_rate}

                # run session
                dice, loss, acc, _, inter, sum_pred, sum_y = sess.run([diceco, cost, accuracy, optimizer, intersection, sum_pred_lbl, sum_y_lbl], feed_dict=feed_dict)

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

                    self.log("【Stage all】Batch_idx: %d, Minibatch Loss: %0.6f, Training Accuracy: %0.5f, Dice Coeffecient: %0.5f " 
                        % (batch_idx, train_loss, train_acc, train_dce))

                    num_train = 0
                    acc_sum_train = 0
                    dce_sum_train = 0
                    loss_sum_train = 0

                if batch_idx % 5000 == 4999:
                    self.learning_rate *= 0.9
                    
                if batch_idx % 500 == 499:
                    
                    num_val = 0
                    acc_sum_val = 0
                    dce_sum_val = 0
                    for val_batch_idx, (image_batch, label_batch) in enumerate(self.data.val_batch()): 
                   
                        
                        feed_dict = {x: image_batch, y: label_batch, weights: weight_vect, lr:self.learning_rate}
                        val_pred_mask, dice, loss, acc = sess.run([pred, diceco, cost, accuracy], feed_dict=feed_dict)

                        n_sample = image_batch.shape[0]
                        num_val += n_sample
                        acc_sum_val += acc * n_sample
                        dce_sum_val += dice * n_sample

                    val_acc = acc_sum_val / num_val
                    val_dce = dce_sum_val / num_val
                    self.log("【Stage all】validation acc: {}".format(val_acc))
                    self.log("【Stage all】validation dice: {}".format(val_dce))

                    # Save the variables to disk.
                    if val_dce > max_val_dce:
                        checkpoint_name = "{}_step_{}_dice_{:.4f}".format(self.model_name+'allstage', batch_idx+1, val_dce)
                        
                        
                        checkpoint_path = os.path.join(self.modelDir, checkpoint_name)
                        saver.save(sess, checkpoint_path)

                        self.log("【Stage all】saving checkpoint to {}".format(checkpoint_path))
                        max_val_dce = val_dce
                        
                    if val_dce > 0.85:
                        return
               
                        