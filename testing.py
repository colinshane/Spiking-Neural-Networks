import tensorflow as tf
import numpy as np
import os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Match GPU IDs to nvidia-smi command
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

np.set_printoptions(linewidth=np.inf)

Vth = 20.
region = np.linspace(-60, 50, num=10000).astype(np.float32)

@tf.custom_gradient
def custom_function(x):

	def grad(dy):
		return dy * (2*x)

	return x**2, grad

@tf.custom_gradient
def spike_function(V):

	def grad(dy):
		return dy * tf.nn.relu(1-tf.abs((V-Vth)/Vth))

	return tf.cast(V > Vth, tf.float32), grad

# Make a variable, run it through the custom op
x = tf.get_variable('x', initializer=region)

y = custom_function(x)

z = spike_function(x)

# Make gradients
g1 = tf.gradients(y, [x])
g2 = tf.gradients(z, [x])

# Start a TensorFlow session, initialize variables, train
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	x, y, z, g1, g2 = sess.run([x, y, z, g1, g2])

	# print('inputs:\t', x)
	
	# print('\nsquare output:  ', y)
	# print('square gradients:', g1[0])


	# print('\nspike output:  ', z)
	# print('spike gradients:', g2[0])
	
	plt.plot(region, z, label='Spike Output')
	plt.plot(region, g2[0], label='Spike Gradient')
	plt.axvline(Vth, ls='--', c='k', label='Threshold')
	plt.axvline(-50, ls=':', c='k', label='Reset Voltage')

	plt.xlabel('Membrane Voltage (mV)')
	plt.ylabel('Spike | Gradient')
	plt.title('Spiking Neuron Behavior and Gradient Response')
	plt.xlim(region.min(), region.max())
	plt.legend()
	plt.savefig('./savedir/spike_gradient.png')