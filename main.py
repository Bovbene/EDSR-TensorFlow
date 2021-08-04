# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 16:17:07 2021

@author: 月光下的云海
"""

# from DLL.Valuation import psnr as psnr
# import cv2 as cv
# from DLL.utils import Show
import argparse
# import numpy as np
# from PIL import Image
# from DLL.utils import DegradeFilter

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = 'EDSR', help=['SrOp','CKP','SLSR','SLKPSR','SRCNN','VDSR'])
parser.add_argument('--is_train', type=bool, default = True, help='Train or test.')
parser.add_argument('--scale', type=int, default = 4, help='Scale Factor')
parser.add_argument('--epoch', type = int, default = 20, help = "The training epochs.")
parser.add_argument('--f1', type = str, default = './DATABASE/DIV2K/DIV2K_train_HR/*.png')
parser.add_argument('--f2', type = str, default = './DATABASE/Set5/*.bmp')
args = parser.parse_args()

if __name__ == '__main__':
    
    exec( "from MODEL.%s import %s as Model"%(args.model,args.model) )
    model = Model(scale = args.scale,epoch = args.epoch)
    if args.is_train:
        print( model.train_scale(args.f1,args.f2) )
    else:
        model.eval(args.f2)