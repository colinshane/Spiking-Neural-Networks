# Model modules
from parameters import *
import stimulus
import AdamOpt

# Required packages
import tensorflow as tf
import numpy as np
import pickle
import os, sys, time

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Match GPU IDs to nvidia-smi command
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class Model:

	def __init__(self):

		self.x = input_data
		self.y = output_data
		self.m = mask_data

		self.run_model()
		self.optimize()

	def run_model(self):

		W = tf.get_variable('W', initializer=tf.zeros(par['n_input'], par['n_output']))
		self.y_hat = tf.stack([W[t,...] @ self.x for t in range(par['num_time_steps'])], axis=0)

	def optimize(self):

		adam_optimizer = AdamOpt.AdamOpt(tf.trainable_variables(), learning_rate=par['learning_rate'])

		self.task_loss = tf.reduce_mean(self.m*tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_hat, labels=self.output_data))

		# Compute gradients
		self.train = adam_optimizer.compute_gradients(self.task_loss)


def main(gpu_id=None):
	""" Run training """

	# Isolate requested GPU
	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	# Reset TensorFlow graph
	tf.reset_default_graph()

	# Define placeholders
	x = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'], par['n_input']], 'stim')
	y = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'], par['n_output']], 'out')
	m = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size']], 'mask')

	# Set up stimulus and recording
	stim = stimulus.Stimulus()

	# Start TensorFlow session
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8) if gpu_id == '0' else tf.GPUOptions()
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		# Select CPU or GPU
		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			model = Model(x, y, m)

		# Initialize variables and start timer
		sess.run(tf.global_variables_initializer())
		t_start = time.time()

		# Begin training loop, iterating over tasks
		for i in range(par['n_train_batches']):

			# Generate a batch of stimulus data for training
			trial_info = stim.make_batch()

			# Put together the feed dictionary
			feed_dict = {x:trial_info['neural_input'], y:trial_info['desired_output'], m:trial_info['train_mask']}

			# Run the model
			_, task_loss, output, state, spike = \
				sess.run([model.train, model.task_loss, \
				model.output, model.h, model.h_out], feed_dict=feed_dict)











if __name__ == '__main__':
	try:
		if len(sys.argv) > 1:
			main(sys.argv[1])
		else:
			main()
	except KeyboardInterrupt:
		quit('Quit by KeyboardInterrupt.')