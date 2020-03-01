import os
import numpy as np
from PIL import Image

class Dataset:
    
    train_img = None
    train_lbl = None
    val_img = None
    val_lbl = None
    test_img = None
    test_lbl = None
    batch_size = None
    max_epoch = None
    
    
    def split_dataset(self):
        all_list = os.listdir(self.dataroot)
        n = len(all_list)
        np.random.shuffle(all_list)
        f1 = open(self.train_file, 'w')
        f2 = open(self.val_file, 'w')
        f3 = open(self.test_file, 'w')
        for i, name in enumerate(all_list):
            if 'txt' in name:
                continue
            if (i <= n/3*2):
                print(name, file=f1)
            elif (i <= n/6*5):
                print(name, file=f2)
            else:
                print(name, file=f3)
        f1.close()
        f2.close()
        f3.close()

    def load_data_to_list(self, file, datapath):
        imglist = []
        lbllist = []
        with open(file, 'r') as f:
            folderlist = f.read().split()
        # print(folderlist)
        for folder in folderlist:
            imgnamelist = os.listdir(os.path.join(datapath, folder))
            # print(imgnamelist)
            for imgname in imgnamelist:
                if 'mask' in imgname:
                    continue
                imgindex = imgname.split('.')[0]
                # print(imgindex, imgname)
                img = Image.open(os.path.join(datapath, folder, imgindex)+'.tif')
                imglist.append(np.array(img))
                lbl = Image.open(os.path.join(datapath, folder, imgindex)+'_mask.tif')
                lblarray = np.array(lbl)/255
                
                lbl2 = np.zeros([256, 256, 2])
                
                
                lbl2[:, :, 0] = lblarray
                lbl2[:, :, 1] = 1-lblarray

                lbllist.append(np.array(lbl2))

        return np.array(imglist)/255, np.array(lbllist)
    
    
    def __init__(self, dataroot, BATCH_SIZE=4, MAX_EPOCH=100):
        self.n_classes = 2
        self.dataroot = dataroot
        
        self.train_file = os.path.join(self.dataroot, 'train.txt')
        self.test_file = os.path.join(self.dataroot, 'test.txt')
        self.val_file = os.path.join(self.dataroot, 'val.txt')
        self.split_dataset()
        
        
        self.train_img, self.train_lbl = self.load_data_to_list(self.train_file, self.dataroot)
        self.val_img, self.val_lbl = self.load_data_to_list(self.val_file, self.dataroot)
        self.test_img, self.test_lbl = self.load_data_to_list(self.test_file, self.dataroot)
        
        
        self.batch_size = BATCH_SIZE
        self.max_epoch = MAX_EPOCH
        
        
    def augment_sample(self, image, label):

        image = image
        label = label
        
        return(image, label)
    
    def get_class_weight(self):
        return self.generate_class_weight(self.train_lbl)
    
    def generate_class_weight(self, labels):
        
        arg_labels = np.argmax(labels, axis = -1)
        class_weights = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            class_weights[i] = 1 / np.mean(arg_labels == i) ** 0.3
        class_weights /= np.sum(class_weights)
        """
        class_weights = np.ones(self.n_classes)
        """
        return class_weights

    def train_batch(self):
        for img, lbl in self.generate_batch(self.train_img, self.train_lbl, self.max_epoch):
            yield img, lbl
    def val_batch(self):
        for img, lbl in self.generate_batch(self.val_img, self.val_lbl, 1):
            yield img, lbl
    def test_batch(self):
        for img, lbl in self.generate_batch(self.test_img, self.test_lbl, 1):
            yield img, lbl
    
    
    def generate_batch(self, images, labels, epochs):
        for samples in self.generate_samples(len(images), epochs):

            image_batch = images[samples]
            label_batch = labels[samples]
            """
            for i in range(image_batch.shape[0]):
                image_batch[i], label_batch[i] = self.augment_sample(image_batch[i], label_batch[i])
            """
            yield(image_batch, label_batch)

    def generate_samples(self, n_samples, epochs):

        n_batches = n_samples/self.batch_size
        for _ in range(epochs):
            sample_ids = np.random.permutation(n_samples)
            for i in range(int(n_batches)):
                inds = slice(i*self.batch_size, (i+1)*self.batch_size)
                yield sample_ids[inds]   