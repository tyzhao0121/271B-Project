# importing needed libraries. 
# you can pip install all of them.
import os

from medpy.io import load
import numpy as np

from tensorflow.contrib.layers import conv2d, conv2d_transpose, max_pool2d, batch_norm

import tensorflow as tf

# Defining parameters

dataPath = '../data/LPBA40/' # where data is
modelPath = '../model/' # where to save the model

# what is loss function, the current options in this notebook are:
# cross_entropy: apply cross entropy on each pixle separately and avrage them on slice
# weighted_cross_entropy: apply cross entropy on each pixle separately and weighted average them on slice based on 
#                         the ratio of classes in each slice
# dice: apply dice coefficient on each slice and minimize 1-dice
# Tverskey: not implemented in this notebook. very useful for highly imblanced data (like 3d MS lesion detection)
loss_method = 'weighted_cross_entropy' # what is loss function, the 

batch_size = 2
display_step = 20

# Network Parameters
tf.reset_default_graph()
width = 256
height = 256
n_channels = 2 # image and probability map
n_classes = 2 # total classes (brain, non-brain)

# total number of slices we are going to train on. Not the best implementation though.
NumberOfSamples = 12
NumberOfSlices = 124


# generate batches during training. one can use keras and forget about this function.
def generate_batch():
    for samples in generate_samples():
        image_batch = images[samples]
        label_batch = labels[samples]
        for i in range(image_batch.shape[0]):
            image_batch[i], label_batch[i] = augment_sample(image_batch[i], label_batch[i])
        yield(image_batch, label_batch)

# choose random slices:
def generate_samples():
    n_samples = NumberOfSamples * NumberOfSlices
    n_epochs = 1000
    n_batches = n_samples/batch_size
    for _ in range(n_epochs):
        sample_ids = np.random.permutation(n_samples)
        for i in range(int(n_batches)):
            inds = slice(i*batch_size, (i+1)*batch_size)
            yield sample_ids[inds]

# you want to add augmentation? (rotation, translation, etc). Do it on_fly! write your augmentation function here:
# right now: do nothing for augmentation! :)
def augment_sample(image, label):

    image = image
    label = label
    
    return(image, label)

# design you model here but first be sure to reset tensorflow graph.
# Unet:
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
    
#######################Train###################################
# To use Auto-Context algorithm the number of steps should be more than 1:
for step in [0]:#xrange(1,4):
    # Load train data and the labels:
    # If it's the first step, the posterior probability is defiend as 0.5
    # Otherwise, the posterior probability will be loaded using output of the previous step.
    images = np.zeros(
        (
            NumberOfSamples*NumberOfSlices,
            width,
            height,
            n_channels,
        )
    )
    labels = np.zeros(
        (
            NumberOfSamples*NumberOfSlices,
            width,
            height,
            n_classes,
        )
    )
    inputCounter = 0
    for f in os.listdir(dataPath+'Images/'):
        if "img" in f:
            print(f)
            inputCounter += 1
            image_data, image_header = load(dataPath+'Images/'+f)
            imageDim = np.shape(image_data)
            image_data_labels, image_header_labels = load(dataPath+'Labels/'+f[:10]+'.brain.mask.img.gz')
            image_data_labels = np.clip(image_data_labels, 0, 1)
            if step > 0:
                image_data_labels_posterior = np.load(dataPath+'G1Posterior_Unet_2/'+str(step-1)+f[:-4]+'.npy')
            else:
                image_data_labels_posterior = np.zeros_like(image_data_labels) + 0.5

            temp = np.swapaxes(image_data,0,1)
            temp2 = np.swapaxes(image_data_labels_posterior,0,1)
            tempL = np.swapaxes(image_data_labels,0,1)
            images[(inputCounter-1)*NumberOfSlices:(inputCounter)*NumberOfSlices,:,:,0] = temp
            images[(inputCounter-1)*NumberOfSlices:(inputCounter)*NumberOfSlices,:,:,1] = temp2*128
            labels[(inputCounter-1)*NumberOfSlices:(inputCounter)*NumberOfSlices,:,:,0] = tempL
            labels[(inputCounter-1)*NumberOfSlices:(inputCounter)*NumberOfSlices,:,:,1] = 1-tempL

            ######################################
            
    # x: place holder for the input image.
    # y: place holder for the labels.
    # lr : place holder for learning rate. to change the learning rate as we move forward. 
    # weights: used in weighted_cross_entropy.
    x = tf.placeholder(tf.float32, [None, width, height, n_channels])
    y = tf.placeholder(tf.float32, [None, width, height, n_classes])
    lr = tf.placeholder(tf.float32)
    weights = tf.placeholder(tf.float32, [batch_size*width*height])
    
    #define model:
    pred = Unet(x)
    
    # Define loss and optimizer
    pred_reshape = tf.reshape(pred, [batch_size * width * height, n_classes])
    y_reshape = tf.reshape(y, [batch_size * width * height, n_classes])

    if loss_method == 'cross_entropy':
        cost = tf.losses.softmax_cross_entropy(onehot_labels = y , logits = pred)

    elif loss_method == 'weighted_cross_entropy':
        cost = tf.losses.softmax_cross_entropy(onehot_labels = y_reshape , logits = pred_reshape, weights=weights)

    elif loss_method == 'dice':
        intersection = tf.reduce_sum(pred_reshape * y_reshape)
        cost = -(2 * intersection + 1)/(tf.reduce_sum(pred_reshape) + tf.reduce_sum(y_reshape) + 1)

    else:
        raise NotImplementedError
        
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, -1), tf.argmax(y, -1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
     # Initializing the variables
    init = tf.global_variables_initializer()
    arg_labels = np.argmax(labels, axis = -1)
    class_weights = np.zeros(n_classes)
    for i in range(n_classes):
        class_weights[i] = 1 / np.mean(arg_labels == i) ** 0.3
    class_weights /= np.sum(class_weights)

    saver = tf.train.Saver()
    model_path = os.path.join(modelPath, 'LPBA_Unet_'+str(step)+'.ckpt')

    sess = tf.Session()
    sess.run(init)
    
    learning_rate = 0.00001       
    for step2, (image_batch, label_batch) in enumerate(generate_batch()):            
        label_vect = np.reshape(np.argmax(label_batch, axis=-1), [batch_size * width * height])
        weight_vect = class_weights[label_vect]
        # Fit training using batch data
        feed_dict = {x: image_batch, y: label_batch, weights: weight_vect, lr:learning_rate}
        loss, acc, _ = sess.run([cost, accuracy, optimizer], feed_dict=feed_dict)
        if step2 % display_step == 0:
            print("Step %d, Minibatch Loss=%0.6f , Training Accuracy=%0.5f " 
                  % (step2, loss, acc))

            # Save the variables to disk.
            saver.save(sess, model_path)
        if step2 % 2000 == 0:
            learning_rate *= 0.9
        
    ##############################Test#########################
    for f in os.listdir(dataPath+'Images/'):
        if "img" in f:
            print(f)
            image_data, image_header = load(dataPath+'Images/'+f)
            imageDim = np.shape(image_data)
            image_data_labels, image_header_labels = load(dataPath+'Labels/'+f[:10]+'.brain.mask.img.gz')
            if step > 0:
                image_data_labels_posterior = np.load(dataPath+'Posterior/'+str(step-1)+f[:-4]+'.npy')
            else:
                image_data_labels_posterior = np.zeros_like(image_data_labels) + 0.5
            image_data_labels = np.clip(image_data_labels, 0, 1)
            temp = np.swapaxes(image_data,0,1)
            temp2 = np.swapaxes(image_data_labels_posterior,0,1)
            Pmask = np.zeros_like(temp)
            ProbRes = np.zeros_like(temp)
            for z in range(0, 124):
                if z % 2 == 0:
                    if z == 124-1:
                        image_batch2 = np.zeros((2,width,height,2), dtype=np.float32)
                        image_batch2[0,:,:,0] = temp[z-1,:,:]
                        image_batch2[1,:,:,0] = temp[z,:,:]
                        image_batch2[0,:,:,1] = temp2[z-1,:,:]*128
                        image_batch2[1,:,:,1] = temp2[z,:,:]*128
                        out = sess.run(tf.nn.softmax(pred_reshape), feed_dict={x: image_batch2})
                        _out = np.reshape(out, (2, width, height, 2)) 
                        resArr = np.asarray(_out)
                        output_image = np.argmax(_out, axis=3)
                        Pmask[z-1,:,:] = 1-output_image[0,:,:]
                        Pmask[z,:,:] = 1-output_image[1,:,:]
                        ProbRes[z-1,:,:] = resArr[0,:,:,1] 
                        ProbRes[z,:,:] = resArr[1,:,:,1]
                    else:
                        image_batch2 = np.zeros((2,width,height,2), dtype=np.float32)
                        image_batch2[0,:,:,0] = temp[z,:,:]
                        image_batch2[1,:,:,0] = temp[z+1,:,:]
                        image_batch2[0,:,:,1] = temp2[z,:,:]*128
                        image_batch2[1,:,:,1] = temp2[z+1,:,:]*128
                        out = sess.run(tf.nn.softmax(pred_reshape), feed_dict={x: image_batch2})
                        _out = np.reshape(out, (2, width, height, 2))      
                        resArr = np.asarray(_out)
                        output_image = np.argmax(_out, axis=3)
                        Pmask[z,:,:] = 1-output_image[0,:,:]
                        Pmask[z+1,:,:] = 1-output_image[1,:,:]
                        ProbRes[z,:,:] = resArr[0,:,:,1] 
                        ProbRes[z+1,:,:] = resArr[1,:,:,1]
            temp2 = np.swapaxes(Pmask,0,1)
            ProbRes2 = np.swapaxes(ProbRes,0,1)  
            np.save(dataPath+'Posterior/'+str(step)+f[:-4],ProbRes2)
            tp = np.sum(np.multiply(temp2,image_data_labels))
            tn = np.sum(np.multiply((1-temp2),(1-image_data_labels)))
            fp = np.sum(np.multiply(temp2,(1-image_data_labels)))
            fn = np.sum(np.multiply((1-temp2),image_data_labels))
            print(2*tp/(2*tp+fp+fn))
            print(tp/(tp+fn))
            print(tn/(tn+fp))

for f in os.listdir(dataPath+'Images/'):
    if "img" in f:
        print(f)
        image_data, image_header = load(dataPath+'Images/'+f)
        imageDim = np.shape(image_data)
        image_data_labels, image_header_labels = load(dataPath+'Labels/'+f[:10]+'.brain.mask.img.gz')
        if step > 0:
            image_data_labels_posterior = np.load(dataPath+'Posterior/'+str(step-1)+f[:-4]+'.npy')
        else:
            image_data_labels_posterior = np.zeros_like(image_data_labels) + 0.5
        image_data_labels = np.clip(image_data_labels, 0, 1)
        temp = np.swapaxes(image_data,0,1)
        temp2 = np.swapaxes(image_data_labels_posterior,0,1)
        Pmask = np.zeros_like(temp)
        ProbRes = np.zeros_like(temp)
        for z in range(0, 124):
            if z % 2 == 0:
                if z == 124-1:
                    image_batch2 = np.zeros((2,width,height,2), dtype=np.float32)
                    image_batch2[0,:,:,0] = temp[z-1,:,:]
                    image_batch2[1,:,:,0] = temp[z,:,:]
                    image_batch2[0,:,:,1] = temp2[z-1,:,:]*128
                    image_batch2[1,:,:,1] = temp2[z,:,:]*128
                    out = sess.run(tf.nn.softmax(pred_reshape), feed_dict={x: image_batch2})
                    _out = np.reshape(out, (2, width, height, 2)) 
                    resArr = np.asarray(_out)
                    output_image = np.argmax(_out, axis=3)
                    Pmask[z-1,:,:] = 1-output_image[0,:,:]
                    Pmask[z,:,:] = 1-output_image[1,:,:]
                    ProbRes[z-1,:,:] = resArr[0,:,:,1] 
                    ProbRes[z,:,:] = resArr[1,:,:,1]
                else:
                    image_batch2 = np.zeros((2,width,height,2), dtype=np.float32)
                    image_batch2[0,:,:,0] = temp[z,:,:]
                    image_batch2[1,:,:,0] = temp[z+1,:,:]
                    image_batch2[0,:,:,1] = temp2[z,:,:]*128
                    image_batch2[1,:,:,1] = temp2[z+1,:,:]*128
                    out = sess.run(tf.nn.softmax(pred_reshape), feed_dict={x: image_batch2})
                    _out = np.reshape(out, (2, width, height, 2))      
                    resArr = np.asarray(_out)
                    output_image = np.argmax(_out, axis=3)
                    Pmask[z,:,:] = 1-output_image[0,:,:]
                    Pmask[z+1,:,:] = 1-output_image[1,:,:]
                    ProbRes[z,:,:] = resArr[0,:,:,1] 
                    ProbRes[z+1,:,:] = resArr[1,:,:,1]
        temp2 = np.swapaxes(Pmask,0,1)
        ProbRes2 = np.swapaxes(ProbRes,0,1)  
        np.save(dataPath+'Posterior/'+str(step)+f[:-4],ProbRes2)
        tp = np.sum(np.multiply(temp2,image_data_labels))
        tn = np.sum(np.multiply((1-temp2),(1-image_data_labels)))
        fp = np.sum(np.multiply(temp2,(1-image_data_labels)))
        fn = np.sum(np.multiply((1-temp2),image_data_labels))
        print(2*tp/(2*tp+fp+fn))
        print(tp/(tp+fn))
        print(tn/(tn+fp))