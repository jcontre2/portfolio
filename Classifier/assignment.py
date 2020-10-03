from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random
import time

class Model(tf.keras.Model):
	def __init__(self):
		super(Model, self).__init__()

		self.batch_size = 100
		self.num_classes = 2
		self.drop_rate = 0.3
		self.learning_rate =.001
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        ##Here we create all the weight and bias variables
		self.W = {}
		self.B = {}
		self.W["conv_1"] = tf.Variable(tf.random.truncated_normal([5,5,3,16], stddev=0.1))
		self.B["conv_1"] = tf.Variable(tf.random.truncated_normal([16], stddev=0.1))
		self.W["conv_2"] = tf.Variable(tf.random.truncated_normal([5,5,16,20], stddev=0.1))
		self.B["conv_2"] = tf.Variable(tf.random.truncated_normal([20], stddev=0.1))
		self.W["conv_3"] = tf.Variable(tf.random.truncated_normal([5,5,20,20], stddev=0.1))
		self.B["conv_3"] = tf.Variable(tf.random.truncated_normal([20], stddev=0.1))
		self.W["dense_1"] = tf.Variable(tf.random.truncated_normal([1280,64], stddev=0.1))
		self.B["dense_1"] = tf.Variable(tf.random.truncated_normal([64], stddev=0.1))
		self.W["dense_2"] = tf.Variable(tf.random.truncated_normal([64,32], stddev=0.1))
		self.B["dense_2"] = tf.Variable(tf.random.truncated_normal([32], stddev=0.1))
		self.W["dense_3"] = tf.Variable(tf.random.truncated_normal([32,2], stddev=0.1))
		self.B["dense_3"] = tf.Variable(tf.random.truncated_normal([2], stddev=0.1))

	def call(self, inputs, is_testing=False):
		"""
		Runs a forward pass on an input batch of images.
		:param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
		:param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
		:return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
		"""
		# shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
		# shape of filter = (filter_height, filter_width, in_channels, out_channels)
		# shape of strides = (batch_stride, height_stride, width_stride, channels_stride)
		conv_1 = tf.nn.conv2d(inputs, self.W["conv_1"], [1,1,1,1], padding="SAME")
		conv_1_with_bias = tf.nn.bias_add(conv_1, self.B["conv_1"])

		mean_1, var_1 = tf.nn.moments(conv_1_with_bias, [0, 1, 2])
		norm_1 = tf.nn.batch_normalization(conv_1_with_bias,mean_1,var_1,None,None,self.learning_rate)

		relu_1 = tf.nn.relu(norm_1)

		pool_1 = tf.nn.max_pool(relu_1,3,[1,2,2,1],padding="SAME")

		conv_2 = tf.nn.conv2d(pool_1, self.W["conv_2"], [1,1,1,1], padding="SAME")
		conv_2_with_bias = tf.nn.bias_add(conv_2, self.B["conv_2"])

		mean_2, var_2 = tf.nn.moments(conv_2_with_bias, [0, 1, 2])
		norm_2 = tf.nn.batch_normalization(conv_2_with_bias,mean_2,var_2,None,None,self.learning_rate)

		relu_2 = tf.nn.relu(norm_2)

		pool_2 = tf.nn.max_pool(relu_2,3,[1,2,2,1],padding="SAME")

        #used for testing sutom cov2D function
		if is_testing:
			print("My Conv2d Begins..")
			conv_3 = conv2d(pool_2, self.W["conv_3"], [1,1,1,1], padding="SAME")
		else:
			conv_3 = tf.nn.conv2d(pool_2, self.W["conv_3"], [1,1,1,1], padding="SAME")

		conv_3_with_bias = tf.nn.bias_add(conv_3, self.B["conv_3"])

		mean_3, var_3 = tf.nn.moments(conv_3_with_bias, [0, 1, 2])
		norm_3 = tf.nn.batch_normalization(conv_3_with_bias,mean_3,var_3,None,None,self.learning_rate)

		relu_3 = tf.nn.relu(norm_3)

		reshaped_layer = self.layer_reshape(relu_3)

		dense_1 = tf.nn.relu(tf.matmul(reshaped_layer,self.W["dense_1"])+self.B["dense_1"])

		drop_1 = tf.nn.dropout(dense_1,self.drop_rate)

		dense_2 = tf.nn.relu(tf.matmul(drop_1,self.W["dense_2"])+self.B["dense_2"])

		drop_2 = tf.nn.dropout(dense_2,self.drop_rate)

		logits = tf.nn.relu(tf.matmul(drop_2,self.W["dense_3"])+self.B["dense_3"])

		return logits


	def layer_reshape(self,layer):
		layer_shape = layer.shape.as_list()
		new_layer_size = layer_shape[-1]*layer_shape[-2]*layer_shape[-3]
		return tf.reshape(layer, [-1, new_layer_size])

	def loss(self, logits, labels):
		"""
		Calculates the model cross-entropy loss after one forward pass.
		:param logits: during training, a matrix of shape (batch_size, self.num_classes)
		containing the result of multiple convolution and feed forward layers
		Softmax is applied in this function.
		:param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
		:return: the loss of the model as a Tensor
		"""
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels,logits))

	def accuracy(self, logits, labels):
		"""
		Calculates the model's prediction accuracy by comparing
		logits to correct labels

		:param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
		    containing the result of multiple convolution and feed forward layers
		:param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)
		:return: the accuracy of the model as a Tensor
		"""
		correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
		return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
	'''
	Trains the model on all of the inputs and labels for one epoch

	:param model: the initialized model to use for the forward pass and backward pass
	:param train_inputs: train inputs (all inputs to use for training),
	   shape (num_inputs, width, height, num_channels)
	:param train_labels: train labels (all labels to use for training),
	   shape (num_labels, num_classes)
	:return: None
	'''

	inputs, labels = shuffle(train_inputs,train_labels)
	tf.image.random_flip_left_right(inputs)
	for i in range(0,len(labels),model.batch_size):
		with tf.GradientTape() as tape:
			logits = model.call(inputs[i:i+model.batch_size])
			loss = model.loss(logits,labels[i:i+model.batch_size])
		gradients = tape.gradient(loss,model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def shuffle(inputs,labels):
	indices = np.arange(len(labels))
	np.random.shuffle(indices)
	return tf.gather(inputs,indices),tf.gather(labels,indices)

def test(model, test_inputs, test_labels):
	"""
	Tests the model on the test inputs and labels.

	:param test_inputs: test data (all images to be tested),
	   shape (num_inputs, width, height, num_channels)
	:param test_labels: test labels (all corresponding labels),
	   shape (num_labels, num_classes)
	:return: test accuracy
	"""
	logits = model.call(test_inputs,True)
	return model.accuracy(logits,test_labels)

def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
	"""
	Uses Matplotlib to visualize the results of our model.
	:param image_inputs: image data from get_data(), limited to 10 images, shape (10, 32, 32, 3)
	:param probabilities: the output of model.call(), shape (10, num_classes)
	:param image_labels: the labels from get_data(), shape (10, num_classes)
	:param first_label: the name of the first class, "dog"
	:param second_label: the name of the second class, "cat"


	:return: doesn't return anything, a plot should pop-up
	"""
	predicted_labels = np.argmax(probabilities, axis=1)
	num_images = image_inputs.shape[0]

	fig, axs = plt.subplots(ncols=num_images)
	fig.suptitle("PL = Predicted Label\nAL = Actual Label")
	for ind, ax in enumerate(axs):
			ax.imshow(image_inputs[ind], cmap="Greys")
			pl = first_label if predicted_labels[ind] == 0.0 else second_label
			al = first_label if np.argmax(image_labels[ind], axis=0) == 0 else second_label
			ax.set(title="PL: {}\nAL: {}".format(pl, al))
			plt.setp(ax.get_xticklabels(), visible=False)
			plt.setp(ax.get_yticklabels(), visible=False)
			ax.tick_params(axis='both', which='both', length=0)
	plt.show()


def main():
    #preprocess cifar dataset
	train_inputs,train_labels = get_data("CIFAR_data_compressed/train",3,5)
	test_inputs,test_labels = get_data("CIFAR_data_compressed/test",3,5)

    #run model
	model = Model()
	for i in range(25):
		holder = time.time()
		train(model,train_inputs,train_labels)


if __name__ == '__main__':
	main()
