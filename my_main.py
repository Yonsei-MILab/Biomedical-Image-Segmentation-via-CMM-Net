"""
Created on Monday February 03 2020

@author: Mohammed Al-masni
"""

import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3' 

###############################
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils, to_categorical,plot_model

from sklearn import preprocessing
from keras.models import Model, model_from_json

from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import backend
import cv2

from models.pspnet import vgg_pspnet
from models.RMSPP_unet import RMSPP_unet
from models.RMSPP_unet_retinal import RMSPP_unet_retinal
from models.RMSPP_unet_brain import RMSPP_unet_brain

from predict import predict
from predict import predict_multiple
from train import train
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6'#'0,1,2,3,4'
##os.environ['TF_KERAS'] = '1'
#------------------------------------------------------------------------------
Train = 1 # True False    
Test  = 1 # True False

epoch = 100
learningRate = 0.0001 # 0.0001
optimizer = Adam(lr=learningRate)
batch_size = 20 #20 #2#8
# Size of images:--> Skin:192x256, RetinalVessels/DRIVE/ : 128x128, brain: 192x192
Height = 192#128#192
Width  = 256#128#192
n_classes   = 2 # binary classification: 0: tissue, 1: lesion
num_train_data = 48000

train_data_path  = '.../ISIC2017/train/image/'  
train_GT_path    = '.../ISIC2017/train/label/'
valid_data_path  = '.../ISIC2017/validation/image/'
valid_GT_path    = '.../ISIC2017/validation/label/'
test_data_path   = '.../ISIC2017/test/image/'
test_GT_path     = '.../ISIC2017/test/label/'

Prediction_path  = '/../Predictions/'
Weights_path     = '/Weights/RMSPP_UNet_'

# ======================================================================================================
def my_main():
	#model = vgg_pspnet(n_classes=n_classes ,  input_height=Height, input_width=Width)    
	
	model = RMSPP_unet(n_classes=n_classes ,  input_height=Height, input_width=Width)
	#model = RMSPP_unet_retinal(n_classes=n_classes ,  input_height=Height, input_width=Width)
	#model = RMSPP_unet_brain(n_classes=n_classes ,  input_height=Height, input_width=Width)   
	    
	if Train:
		print('Generating DL Model...')
			
		model.summary()
		
		train(
			model,
			train_images =  train_data_path,
			train_annotations = train_GT_path,
			input_height=Height,
			input_width=Width,
			n_classes=n_classes,
			checkpoints_path = Weights_path,
			epochs=epoch,
			batch_size=batch_size,
			validate = True,            
			val_images=valid_data_path,
			val_annotations=valid_GT_path,
			val_batch_size=2,
			steps_per_epoch=num_train_data/batch_size,
			optimizer_name= optimizer,
		)

	if Test:
		predict_multiple(
			checkpoints_path=Weights_path, 
			inp_dir=test_data_path, 
			out_dir=Prediction_path
		)
		# evaluating the model
		IoU_score = model.evaluate_segmentation( inp_images_dir=test_data_path, annotations_dir=test_GT_path )
		print('=================================================')
		print('Model Evaluation')
		print('IoU = ',IoU_score)
		print('=================================================')
		
if __name__ == "__main__":
	my_main()  
	
	
	
