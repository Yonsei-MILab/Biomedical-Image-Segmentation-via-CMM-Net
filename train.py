import argparse
import json
from data_utils.data_loader import image_segmentation_generator, verify_segmentation_dataset
import os
import glob
import six
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import numpy as np
from sklearn.utils import class_weight 
import tensorflow as tf
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)


def find_latest_checkpoint(checkpoints_path, fail_safe=True):

	def get_epoch_number_from_path(path):
		return path.replace(checkpoints_path, "").strip(".")

	# Get all matching files
	all_checkpoint_files = glob.glob(checkpoints_path + ".*")
	# Filter out entries where the epoc_number part is pure number
	all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f).isdigit(), all_checkpoint_files))
	if not len(all_checkpoint_files):
		# The glob list is empty, don't have a checkpoints_path
		if not fail_safe:
			raise ValueError("Checkpoint path {0} invalid".format(checkpoints_path))
		else:
			return None

	# Find the checkpoint file with the maximum epoch
	latest_epoch_checkpoint = max(all_checkpoint_files, key=lambda f: int(get_epoch_number_from_path(f)))
	return latest_epoch_checkpoint
	
# ====================================================
def dice_coef(y_true, y_pred, smooth=1):
	intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
	return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
	return 1-dice_coef(y_true, y_pred)

def train(model,
		  train_images,
		  train_annotations,
		  input_height=None,
		  input_width=None,
		  n_classes=None,
		  verify_dataset=True,
		  checkpoints_path=None,
		  epochs=5,
		  batch_size=2,
		  validate=True,
		  val_images=None,
		  val_annotations=None,
		  val_batch_size=2,
		  auto_resume_checkpoint=False,
		  load_weights=None,
		  steps_per_epoch=512,
		  optimizer_name='adadelta' , do_augment=False , 
		  loss_name='categorical_crossentropy'
		  ):
	#categorical_crossentropy
	from models.all_models import model_from_name
	#from .models.all_models import model_from_name
	# check if user gives model name instead of the model object
	if isinstance(model, six.string_types):
		# create the model from the name
		assert (n_classes is not None), "Please provide the n_classes"
		if (input_height is not None) and (input_width is not None):
			model = model_from_name[model](
				n_classes, input_height=input_height, input_width=input_width)
		else:
			model = model_from_name[model](n_classes)
	
	n_classes = model.n_classes
	input_height = model.input_height
	input_width = model.input_width
	output_height = model.output_height
	output_width = model.output_width
	
	csv_logger = CSVLogger('.../Loss_Acc.csv', append=True, separator=' ')
	checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
	reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min')
	
	if validate:
		assert val_images is not None
		assert val_annotations is not None
		
	
	if optimizer_name is not None:
		model.compile(loss=dice_coef_loss,
					  optimizer=optimizer_name,
					  metrics=[jacard_coef, 'accuracy'])
	
	if checkpoints_path is not None:
		with open(checkpoints_path+"_config.json", "w") as f:
			json.dump({
				"model_class": model.model_name,
				"n_classes": n_classes,
				"input_height": input_height,
				"input_width": input_width,
				"output_height": output_height,
				"output_width": output_width
			}, f)

	if load_weights is not None and len(load_weights) > 0:
		print("Loading weights from ", load_weights)
		model.load_weights(load_weights)

	if auto_resume_checkpoint and (checkpoints_path is not None):
		latest_checkpoint = find_latest_checkpoint(checkpoints_path)
		if latest_checkpoint is not None:
			print("Loading the weights from latest checkpoint ",
				  latest_checkpoint)
			model.load_weights(latest_checkpoint)

	if verify_dataset:
		print("Verifying training dataset") 
		print("Verifying training dataset::")        
		verified = verify_segmentation_dataset(train_images, train_annotations, n_classes)
		assert verified
		if validate:
			print("Verifying validation dataset::")
			verified = verify_segmentation_dataset(val_images, val_annotations, n_classes)
			assert verified

	train_gen = image_segmentation_generator(
		train_images, train_annotations,  batch_size,  n_classes,
		input_height, input_width, output_height, output_width , do_augment=do_augment )

	if validate:
		val_gen = image_segmentation_generator(
			val_images, val_annotations,  val_batch_size,
			n_classes, input_height, input_width, output_height, output_width)

	if not validate:
		for ep in range(epochs):
			print("Starting Epoch # ", ep)
			model.fit_generator(train_gen, steps_per_epoch, epochs=1)
			if checkpoints_path is not None:
				model.save_weights(checkpoints_path + "." + str(ep))
				print("saved ", checkpoints_path + ".model." + str(ep))
			print("Finished Epoch #", ep)
	else:
		for ep in range(epochs):
			print("Starting Epoch # ", ep)
			model.fit_generator(train_gen, steps_per_epoch,
								validation_data=val_gen,
								validation_steps=200,  epochs=1, callbacks=[csv_logger, reduce_lr_loss])
			if checkpoints_path is not None:
				model.save_weights(checkpoints_path + "." + str(ep))
				print("saved ", checkpoints_path + ".model." + str(ep))
			print("Finished Epoch #", ep)
