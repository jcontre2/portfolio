import pickle
import numpy as np
import tensorflow as tf


def unpickle(file):
	"""
	:param file: the file to unpickle
	:return: dictionary of unpickled data
	"""
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


def get_data(file_path, first_class, second_class):
	"""
	Given a file path and two target classes, returns an array of
	normalized inputs (images) and an array of labels.

	:param file_path: file path for inputs and labels, something
	   like 'CIFAR_data_compressed/train'
	:param first_class:  an integer (0-9) representing the first target
	   class in the CIFAR10 dataset, for a cat, this would be a 3
	:param first_class:  an integer (0-9) representing the second target
	   class in the CIFAR10 dataset, for a dog, this would be a 5
	:return: normalized NumPy array of inputs and tensor of labels, where
	   inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and labels
	   has size (num_examples, num_classes)
	"""
	unpickled_file = unpickle(file_path)
	inputs = unpickled_file[b'data']
	labels = unpickled_file[b'labels']
	new_inputs = []
	new_labels = []
	for i in range(len(labels)):
		if labels[i] == first_class:
			new_inputs.append(inputs[i])
			new_labels.append(0)
		if labels[i] == second_class:
			new_inputs.append(inputs[i])
			new_labels.append(1)

	new_inputs = np.transpose(np.reshape(new_inputs, (-1, 3, 32 ,32)), [0,2,3,1])
	new_labels = tf.one_hot(new_labels, depth=2)
	new_inputs = tf.cast(new_inputs/255.0, tf.float32)


	return new_inputs, new_labels
