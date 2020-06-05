import keras
from keras.models import *
from keras.layers import *
from keras import layers
import keras.backend as K

from models.config import IMAGE_ORDERING
from models.model_utils import get_segmentation_model,resize_image



if IMAGE_ORDERING == 'channels_first':
	MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
	MERGE_AXIS = -1


def pool_block(feats, pool_factor):

	if IMAGE_ORDERING == 'channels_first':
		h = K.int_shape(feats)[2]
		w = K.int_shape(feats)[3]
	elif IMAGE_ORDERING == 'channels_last':
		h = K.int_shape(feats)[1]
		w = K.int_shape(feats)[2]

	pool_size = strides = [
		int(np.round(float(h) / pool_factor)),
		int(np.round(float(w) / pool_factor))]

	x = AveragePooling2D(pool_size, data_format=IMAGE_ORDERING, strides=strides, padding='same')(feats)
	x = Conv2D(128, (1, 1), data_format=IMAGE_ORDERING, padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = resize_image(x, strides, data_format=IMAGE_ORDERING)

	return x


def RMSPP_unet_retinal(n_classes, input_height=192, input_width=256):
	
	assert input_height % 32 == 0
	assert input_width % 32 == 0
	
		
	if IMAGE_ORDERING == 'channels_first':
		img_input = Input(shape=(3, input_height, input_width))
	elif IMAGE_ORDERING == 'channels_last':
		img_input = Input(shape=(input_height, input_width, 3))
	
	# Block 1 in Contracting Path
	conv1 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,padding='same', dilation_rate=6)(img_input)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)    
	
	conv1 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=6)(conv1)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)  
	
	pool_factors = [1, 4, 16, 64]#[1, 4, 8]#[1, 2, 4, 8]
	pool_outs = [conv1]

	for p in pool_factors:
		pooled = pool_block(conv1, p)
		pool_outs.append(pooled)

	o1 = Concatenate(axis=MERGE_AXIS)(pool_outs)
	
	o = AveragePooling2D((2, 2), strides=(2, 2))(o1)
	
	# Block 2 in Contracting Path
	conv2 = Conv2D(96, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=5)(o)
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)      
	conv2 = Dropout(0.2)(conv2)
	conv2 = Conv2D(96, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=5)(conv2)
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)   
	
	pool_factors =  [1, 4, 16, 64]#[1, 4, 8]#[1, 2, 4, 8]
	pool_outs = [conv2]

	for p in pool_factors:
		pooled = pool_block(conv2, p)
		pool_outs.append(pooled)

	o2 = Concatenate(axis=MERGE_AXIS)(pool_outs)
	o = AveragePooling2D((2, 2), strides=(2, 2))(o2)    
	
	# Block 3 in Contracting Path
	conv3 = Conv2D(256, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=4)(o)
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)   
	#conv3 = Dropout(0.2)(conv3)
	conv3 = Conv2D(256, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=4)(conv3)
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)  
	
	pool_factors =  [1, 4, 16, 32]#[1, 4, 8]#[1, 2, 4, 8]
	pool_outs = [conv3]

	for p in pool_factors:
		pooled = pool_block(conv3, p)
		pool_outs.append(pooled)

	o3 = Concatenate(axis=MERGE_AXIS)(pool_outs)
	#o = AveragePooling2D((2, 2), strides=(2, 2))(o3)    
	
	
	 # Transition layer between contracting and expansive paths:
	o = AveragePooling2D((2, 2), strides=(2, 2))(o3)
	conv4 = Conv2D(512, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=3)(o)
	conv4 = BatchNormalization()(conv4)
	conv4 = Activation('relu')(conv4)   

	conv4 = Conv2D(512, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=3)(conv4)
	conv4 = BatchNormalization()(conv4)
	conv4 = Activation('relu')(conv4)   
	
		
	# Block 1 in Expansive Path
	up1 = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(conv4)
	up1 = concatenate([up1, o3], axis=MERGE_AXIS)
	deconv1 =  Conv2D(256, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=4)(up1)
	deconv1 = BatchNormalization()(deconv1)
	deconv1 = Activation('relu')(deconv1)    
	#deconv1 = Dropout(0.2)(deconv1)   
	
	deconv1 =  Conv2D(256, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=4)(deconv1)
	deconv1 = BatchNormalization()(deconv1)
	deconv1 = Activation('relu')(deconv1)         
	
	# Block 2 in Expansive Path
	up2 = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(deconv1)    
	up2 = concatenate([up2, o2], axis=MERGE_AXIS)
	deconv2 = Conv2D(96, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=5)(up2)  
	deconv2 = BatchNormalization()(deconv2)
	deconv2 = Activation('relu')(deconv2)
	#deconv2 = Dropout(0.2)(deconv2)       
	
	deconv2 = Conv2D(96, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=5)(deconv2)  
	deconv2 = BatchNormalization()(deconv2)
	deconv2 = Activation('relu')(deconv2)    
	
	# Block 3 in Expansive Path
	up3 = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(deconv2)    
	up3 = concatenate([up3, o1], axis=MERGE_AXIS)
	deconv3 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=6)(up3)  
	deconv3 = BatchNormalization()(deconv3)
	deconv3 = Activation('relu')(deconv3)
	#deconv3 = Dropout(0.2)(deconv3)       
	
	deconv3 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=6)(deconv3)  
	deconv3 = BatchNormalization()(deconv3)
	deconv3 = Activation('relu')(deconv3)    
	
	  
	o = Conv2D(n_classes, (3, 3), data_format=IMAGE_ORDERING, padding='same')(deconv3)
	
	model = get_segmentation_model(img_input, o)
	model.model_name = "RMSPP_unet_retinal"

	return model    
	
 
