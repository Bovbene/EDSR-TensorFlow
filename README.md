# EDSR-TensorFlow
## The TensorFlow code of EDSR.

## ===============================================================================================
The Python code of EDSR
-----------------------------------------------------------------------------------------------
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
### -----------------------------------------------------------------------------------------------
Tip: None
### -----------------------------------------------------------------------------------------------
### Created on Sat May  1 12:21:44 2021
### @author: 月光下的云海(西电博巍)
### Version: Ultimate
## ===============================================================================================
