# -*- coding: utf-8 -*-
"""===============================================================================================
The Python code of EDSR
---------------------------------------------------------------------------------------------------
Class: EDSR
Param: 	self.B = 32 #Number of resBlocks
        self.F = 256 #Number of filters
        self.batch_size = 16 
        self.epochs = 1
        self.scale = scale
        self.lr = 1e-4
        self.scaling_factor = 0.1
        self.PS = 3 * (self.scale*self.scale) #channels x scale^2
        self.is_norm = True
        if self.is_norm:
            self.mean = [103.1545782, 111.561547, 114.35629928]
        else:
            self.mean = [0,0,0]
        self.save_path = './TRAINED_MODEL/'
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.blk_size = 32
---------------------------------------------------------------------------------------------------
Tip: None
---------------------------------------------------------------------------------------------------
Created on Sat May  1 12:21:44 2021
@author: 月光下的云海(西电博巍)
Version: Ultimate
==============================================================================================="""

from .BasicModel import BasicModel
import tensorflow as tf
import numpy as np
import os
from time import time,strftime,localtime
from glob import glob
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from UTILS.util import del_file,setup_logger,plot,flush,tick
import logging
from UTILS.Valuation import ssim,PSNR

class EDSR(BasicModel):
    
    def __init__(self,scale = 4,epoch = 10):
        
        # --Basic Settings--
        self.B = 32 #Number of resBlocks
        self.F = 256 #Number of filters
        self.batch_size = 48 
        self.epochs = epoch
        self.scale = scale
        self.lr = 1e-4
        self.scaling_factor = 0.1
        self.PS = 3 * (self.scale*self.scale) #channels x scale^2
        self.save_path = './TRAINED_MODEL/'
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.blk_size = 32
        self.stride = 16
        # self.c_dim = 3
        self.lr_decay_steps = 5
        self.lr_decay_rate = 0.5
        tf.reset_default_graph()
        self.sess = tf.Session(config=self.config)
        self.build()
        
    def build(self):
        self.xavier = tf.contrib.layers.xavier_initializer()
        self.bias_initializer = tf.constant_initializer(value=0.0)
                     
        # -- Filters & Biases --
        self.resFilters = list()
        self.resBiases = list()
        
        
        for i in range(self.B*2):
            self.resFilters.append( tf.get_variable("EDSR/resFilter%d" % (i), shape=[3,3,self.F,self.F], initializer=self.xavier))
            self.resBiases.append(tf.get_variable(name="EDSR/resBias%d" % (i), shape=[self.F], initializer=self.bias_initializer))
        self.filter_one = tf.get_variable("EDSR/resFilter_one", shape=[3,3,3,self.F], initializer=self.xavier)
        self.filter_two = tf.get_variable("EDSR/resFilter_two", shape=[3,3,self.F,self.F], initializer=self.xavier)
        self.filter_three = tf.get_variable("EDSR/resFilter_three", shape=[3,3,self.F,self.PS], initializer=self.xavier)
        
        self.bias_one = tf.get_variable(shape=[self.F], initializer=self.bias_initializer, name="EDSR/BiasOne")
        self.bias_two = tf.get_variable(shape=[self.F], initializer=self.bias_initializer, name="EDSR/BiasTwo")
        self.bias_three = tf.get_variable(shape=[self.PS], initializer=self.bias_initializer, name="EDSR/BiasThree")
        
    def resBlock(self,x,f_nr):
        y = tf.nn.conv2d(x, filter=self.resFilters[f_nr], strides=[1, 1, 1, 1], padding='SAME')
        y = y+self.resBiases[f_nr]
        y = tf.nn.relu(x)
        y = tf.nn.conv2d(y, filter=self.resFilters[f_nr+1], strides=[1, 1, 1, 1], padding='SAME')
        y = y+self.resBiases[f_nr+1]
        y = y*self.scaling_factor
        out = x+y
        return out
    
    def edsr(self,x):
        # -- Model architecture --

        # first conv
        y = tf.nn.conv2d(x, filter=self.filter_one, strides=[1, 1, 1, 1], padding='SAME')
        y = y+self.bias_one
        y1 = tf.identity(y)
        
        # all residual blocks
        for i in range(self.B):
            y = self.resBlock(y,(i*2))
        
        # last conv
        y = tf.nn.conv2d(y, filter=self.filter_two, strides=[1, 1, 1, 1], padding='SAME')
        y = y+self.bias_two
        y = y+y1
        
        # upsample via sub-pixel, equivalent to depth to space
        y = tf.nn.conv2d(y, filter=self.filter_three, strides=[1, 1, 1, 1], padding='SAME')
        y = y+self.bias_three
        if self.scale == 1: 
            y = tf.identity(y,name = 'EDSR_OUTPUT')
            return y # tf.identity(y,name = 'EDSR')
        y = tf.nn.depth_to_space(y,self.scale,data_format = 'NHWC', name = 'NHWC_output')
        y = tf.identity(y,name = 'EDSR_OUTPUT')
        return y #tf.identity(y,name = 'EDSR')
    
    
    def train(self, imagefolder,validfolder):
        
        setup_logger('base','./TRAINED_MODEL/','train_on_EDSRx{}'.format(self.scale), level=logging.INFO,screen=True, tofile=True)
        logger = logging.getLogger('base')
        
        logger.info("Prepare Data...")
        data = super().input_setup(imagefolder,self.scale,self.blk_size,self.stride)
        if len(data) == 0:
            logger.info("\nCan Not Find Training Data!\n")
            return

        data_dir = super().get_data_dir("./DATABASE/", self.scale)
        data_num = super().get_data_num(data_dir)
        batch_num = data_num // self.batch_size

        images = tf.placeholder(tf.float32, shape = [None,None,None,3], name='images')
        labels = tf.placeholder(tf.float32, shape = [None,None,None,3], name='labels')
        pred = self.edsr(images)
        print(pred)
        counter = 0
        epoch_start = counter // batch_num
        batch_start = counter % batch_num

        logger.info("The parameters volumn is:")
        all_vars = tf.global_variables()
        logger.info(super().count_param(all_vars))
        
        learning_step,loss,learning_rate = super().grenerate_train_op(pred, 
                                                                           labels, 
                                                                           self.lr_decay_steps*batch_num, 
                                                                           self.lr, 
                                                                           self.lr_decay_rate, 
                                                                           var_list = None,
                                                                           loss_function = super().l1_loss)
        
        self.summary = tf.summary.scalar('loss', loss)
        
        tf.global_variables_initializer().run(session=self.sess)

        merged_summary_op = tf.summary.merge_all()
        if os.path.exists('./LOGS/EDSR'):
            del_file("./LOGS/EDSR/")
        summary_writer = tf.summary.FileWriter('./LOGS/EDSR', self.sess.graph)

        test_path = glob(validfolder)
        
        flag = -float("Inf")
        logger.info("Now Start Training EDSR ... ...")
        for ep in range(epoch_start, self.epochs):
            # Run by batch images
            for idx in range(batch_start, batch_num):
                batch_images, batch_labels = super().get_batch(data_dir, data_num, self.batch_size)
                counter += 1

                _, err, lr = self.sess.run([learning_step, loss, learning_rate], feed_dict={images: batch_images, labels: batch_labels})
                plot('Loss', err)
                
                if counter % 10 == 0:
                    flush()
                    logger.info("Epoch: [%4d/%4d], batch: [%6d/%6d], loss: [%.8f], lr: [%.6f], step: [%d]" % ((ep+1), (self.epochs), (idx+1), batch_num, err, lr, counter))
                tick()
                
                if counter % 20 == 0:
                    avg_psnr = 0
                    for p in test_path:
                        input_, label_ = super().get_image(p, self.scale, None)
                        sr = self.sess.run(pred, feed_dict={images: input_})
                        sr = np.squeeze(sr)
                        sr = np.clip(sr, 0, 255)
                        psnr = PSNR(sr, label_[0], self.scale)
                        avg_psnr += psnr
                    avg_psnr = avg_psnr/len(test_path)
                    logger.info("Ave PSNR is:" + str(avg_psnr) )
                    if avg_psnr>flag:
                        logger.info("Saving the better model... ...")
                        constant_graph = graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ["EDSR_OUTPUT"])
                        with tf.gfile.FastGFile(self.save_path+'EDSR_x{}.pb'.format(self.scale), mode='wb') as f:
                            f.write(constant_graph.SerializeToString())
                        flag = avg_psnr
                    summary_str = self.sess.run(merged_summary_op, feed_dict={images: batch_images, labels: batch_labels})
                    summary_writer.add_summary(summary_str, counter)
                    plot('Test PSNR', avg_psnr)
                    flush()
                    tick()
                if counter > 0 and counter == batch_num * self.epochs:
                    avg_psnr = 0
                    for p in test_path:
                        input_, label_ = super().get_image(p, self.scale, None)
                        sr = self.sess.run([pred], feed_dict={images: input_})
                        sr = np.squeeze(sr)
                        sr = np.clip(sr, 0, 255)
                        psnr = PSNR(sr, label_[0], self.scale)
                        avg_psnr += psnr
                    logger.info("Ave PSNR is:" + str(avg_psnr/len(test_path)) )
                    if avg_psnr>flag:
                        logger.info("Saving the better model... ...")
                        constant_graph = graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ["EDSR"])
                        with tf.gfile.FastGFile(self.save_path+'EDSR_x{}.pb'.format(self.scale), mode='wb') as f:
                            f.write(constant_graph.SerializeToString())
                        flag = avg_psnr
                    break

        summary_writer.close()
        logger.info('Train End at'+strftime("%Y-%m-%d %H:%M:%S", localtime())+'The Final Threshold Is:(PSNR:{:.4f}).'.format(flag))
        logger.info("--<The training porcess of EDSR has been completed.>--")
        return "--<The training porcess of EDSR has been completed.>--"
    
    def test(self,image):
        if image is None:
            raise AttributeError("U must input an image path or an image mtx.")
        elif type(image) == type(np.array([])):
            lr_image = image
        elif type(image) == type(''):
            lr_image = super().imread(image)
        else:
            raise AttributeError("U mustn't input an image path and an image mtx concruuently.")
            
        input_ = lr_image[np.newaxis,:]
        pbPath = "./TRAINED_MODEL/EDSR_x{}.pb".format(self.scale)
        with gfile.FastGFile(pbPath,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def,name = '')
        LR_tensor = self.sess.graph.get_tensor_by_name("images:0")
        HR_tensor = self.sess.graph.get_tensor_by_name("EDSR_OUTPUT:0")
        time_ = time()
        result = self.sess.run([HR_tensor], feed_dict={LR_tensor: input_})
        x = np.squeeze(result)
        x = np.clip(x, 0, 255)
        print("Time Elapsed:", time()-time_)
        return np.uint8(x)
    
    def eval(self, validfolder):
        data_set = os.path.split(validfolder)[0]
        data_set = os.path.split(data_set)[-1]
        if not os.path.exists('./RESULT/EDSRx{}/'.format(self.scale)+data_set):
            os.makedirs('./RESULT/EDSRx{}/'.format(self.scale)+data_set)
        print("\nPrepare Data...\n")
        paths = glob(validfolder)
        data_num = len(paths)
        avg_time = 0
        avg_pasn = 0
        avg_ssim = 0
        print("\nNow Start Testing...\n")
        pbPath = "./TRAINED_MODEL/EDSR_x{}.pb".format(self.scale)
        with gfile.FastGFile(pbPath,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def,name = '')
        LR_tensor = self.sess.graph.get_tensor_by_name("images:0")
        HR_tensor = self.sess.graph.get_tensor_by_name("EDSR_OUTPUT:0")
        for idx in range(data_num):
            input_, label_ = super().get_image(paths[idx], self.scale, None)
            fn = os.path.split(paths[idx])[-1]
            fn = os.path.splitext(fn)[0]
            time_ = time()
            result = self.sess.run([HR_tensor], feed_dict={LR_tensor: input_})
            avg_time += time() - time_
            x = np.squeeze(result)
            x = np.clip(x, 0, 255)
            psnr = PSNR(x, label_[0], self.scale)
            avg_pasn += psnr
            issim = ssim(x, label_[0])
            avg_ssim += issim
            print("image: %d/%d, time: %.4f, psnr: %.4f" % (idx, data_num, time() - time_ , psnr))
            if not os.path.isdir(os.path.join(os.getcwd(),"./RESULT/EDSRx{}/".format(self.scale)+data_set)):
                os.makedirs(os.path.join(os.getcwd(),"./RESULT/EDSRx{}/".format(self.scale)+data_set))
            super().imsave(x[:, :, ::-1], "./RESULT/EDSRx{}/".format(self.scale)+data_set+'/' + fn +"_{:.4f}_{:.4f}.png" .format(psnr,issim) )
        print("Avg. Time:", avg_time / data_num)
        print("Avg. PSNR:", avg_pasn / data_num)
        print("Avg. SSIM:", avg_ssim / data_num)
        
        
    
       

