### Authors: Nicolas Y. Masse, Gregory D. Grant

# Model modules
from parameters import *
import stimulus
import AdamOpt

# Required packages
import tensorflow as tf
import numpy as np
from collections import deque
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

	""" Spiking RNN model, supervised learning """

	def __init__(self, input_data, target_data, mask):

		# Load input activity, target data, training mask, etc.
		self.input_data         = tf.unstack(input_data, axis=0)
		self.target_data        = target_data
		self.time_mask          = mask

		# Declare all TensorFlow variables
		self.declare_variables()

		# Build the TensorFlow graph
		self.rnn_cell_loop()

		# Train the model
		self.optimize()

		print('Graph defined.\n')


	def declare_variables(self):
		""" Initialize all required variables """

		var_prefixes = ['W_in', 'W_rnn', 'b_rnn', 'W_out', 'b_out']
		self.var_dict = {}

		with tf.variable_scope('network'):
			for p in var_prefixes:
				self.var_dict[p] = tf.get_variable(p, initializer=par[p+'_init'])


	def rnn_cell_loop(self):
		""" Initialize network state and execute loop through time
			to generate the network outputs """

		# Set up initial state
		self.h_out = [tf.zeros([par['batch_size'],par['n_hidden']])]			# Spike
		self.h = tf.ones([par['batch_size'],par['n_hidden']])					# State
		self.h *= 0.1 if par['cell_type'] == 'rate' else par[par['cell_type']]['V_r']
		self.h = [self.h]
		adapt = par['w_init']*tf.ones([par['batch_size'],par['n_hidden']])

		syn_x = par['syn_x_init']*tf.ones([par['batch_size'], par['n_hidden']]) if par['use_stp'] else None
		syn_u = par['syn_u_init']*tf.ones([par['batch_size'], par['n_hidden']]) if par['use_stp'] else None

		# Apply the EI mask to the recurrent weights
		self.W_rnn_effective = par['EI_matrix'] @ tf.nn.relu(self.var_dict['W_rnn'])

		# Set up latency buffer if being used
		if par['use_latency']:
			self.state_buffer = [tf.zeros([par['batch_size'], par['n_hidden']]) for t in range(par['latency_max'])]
			self.state_buffer = deque(self.state_buffer)
			self.W_rnn_latency = self.W_rnn_effective[tf.newaxis,...] * par['latency_mask']
			self.lat_spike_shape = tf.ones([par['latency_max'], 1, 1])

		# Set up output record
		self.output = []

		y = 0.
		for t in range(par['num_time_steps']):
			self.t = t 		# For latency calculations

			if par['cell_type'] == 'rate':
				raise Exception('Rate cell not yet implemented.')
			elif par['cell_type'] == 'adex':
				spike, state, adapt, syn_x, syn_u = self.AdEx_cell(self.h_out[-1], self.h[-1], adapt, self.input_data[t], syn_x, syn_u)
				y = 0.98*y + 0.02*(spike @ self.var_dict['W_out'] + 0.*self.var_dict['b_out'])

				self.h_out.append(spike)
				self.h.append(state)
				self.output.append(y)

			elif par['cell_type'] == 'lif':
				spike, state, adapt, syn_x, syn_u = self.LIF_cell(self.h_out[-1], self.h[-1], adapt, self.input_data[t], syn_x, syn_u)
				y = 0.95*y + 0.05*spike @ self.var_dict['W_out'] + 0.*self.var_dict['b_out']

				self.h_out.append(spike)
				self.h.append(state)
				self.output.append(y)

		# Stack records
		self.output = tf.stack(self.output, axis=0)
		self.h = tf.stack(self.h, axis=0)
		self.h_out = tf.stack(self.h_out, axis=0)


	def synaptic_plasticity(self, h, syn_x, syn_u):

		if par['use_stp']:
			syn_x += par['alpha_std']*(1-syn_x) - par['stp_mod']*syn_u*syn_x*h
			syn_u += par['alpha_stf']*(par['U']-syn_u) + par['stp_mod']*par['U']*(1-syn_u)*h
			syn_x = tf.minimum(1., tf.nn.relu(syn_x))
			syn_u = tf.minimum(1., tf.nn.relu(syn_u))
			h_post = syn_u*syn_x*h
		else:
			h_post = h

		return h_post, syn_x, syn_u


	def rnn_matmul(self, spike_in):

		if par['use_latency']:
			spike_in = self.lat_spike_shape * spike_in[tf.newaxis,...]
			state_update = tf.unstack(spike_in @ self.W_rnn_latency, axis=0)

			self.state_buffer.rotate(-1)
			self.state_buffer[-1] = tf.zeros_like(self.state_buffer[-1])
			for i, s in enumerate(state_update):
				self.state_buffer[i] += s

			return self.state_buffer[0]

		else:
			return spike_in @ self.W_rnn_effective


	def run_adex(self, V, w, I):

		c = par['adex']
		I = I/c['current_divider']

		# Calculate the new membrane potential
		term1  = I + c['g']*c['D']*tf.exp((V-c['V_T'])/c['D'])
		term2  = w + c['g']*(V-c['E'])
		V_next = V + (c['dt']/c['C'])*(term1-term2)

		# Calculate new adaptation current
		term1  = c['a']*(V-c['E'])
		term2  = w
		w_next = w + (c['dt']/c['tau'])*(term1-term2)

		# Check potential threshold for new spikes
		@tf.custom_gradient
		def do_spike(V):
			def grad(dy): return -0.3*tf.nn.relu(1-tf.abs((V-c['Vth'])/(0.02 + c['Vth'])))
			return tf.cast(V >= c['Vth'], tf.float32), grad

		spike = do_spike(tf.minimum(c['Vth']+1e-9, V_next))

		# Apply spike to membrane and adaptation
		V_new = spike*c['V_r']        + (1-spike)*V_next
		w_new = spike*(w_next+c['b']) + (1-spike)*w_next

		return V_new, w_new, spike


	def run_lif(self, V, w, I):

		V_next = (1-par['lif']['alpha_neuron'])*V + par['lif']['alpha_neuron']*I
		w_next = w

		@tf.custom_gradient
		def do_spike(V):
			def grad(dy): return dy*0.999*tf.nn.relu(1-tf.abs((V/0.04)))
			return tf.cast(V >= par['lif']['Vth'], tf.float32), grad

		spike = do_spike(tf.minimum(par['lif']['Vth']+1e-9, V_next))
		V_new = spike*par['lif']['V_r'] + (1-spike)*V_next
		w_new = spike*w_next + (1-spike)*w_next

		return V_new, w_next, spike


	def LIF_cell(self, spike, V, w, rnn_input, syn_x, syn_u):

		# Apply synaptic plasticity
		spike_post, syn_x, syn_u = self.synaptic_plasticity(spike, syn_x, syn_u)

		I = rnn_input @ self.var_dict['W_in'] + self.rnn_matmul(spike_post) + 0*self.var_dict['b_rnn']
		V, w, spike = self.run_lif(V, w, I)

		return spike, V, w, syn_x, syn_u


	def AdEx_cell(self, spike, V, w, rnn_input, syn_x, syn_u):

		# Apply synaptic plasticity
		spike_post, syn_x, syn_u = self.synaptic_plasticity(spike, syn_x, syn_u)

		I = rnn_input @ self.var_dict['W_in'] + self.rnn_matmul(spike_post) + self.var_dict['b_rnn']
		V, w, spike = self.run_adex(V, w, I)

		return spike, V, w, syn_x, syn_u


	def optimize(self):
		""" Calculate losses and apply corrections to model """

		# Set up optimizer
		adam_optimizer = AdamOpt.AdamOpt(tf.trainable_variables(), learning_rate=par['learning_rate'])

		# Calculate losses
		self.task_loss = tf.reduce_mean(self.time_mask * \
				tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.target_data))

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
	data_record = {n:[] for n in ['iter', 'acc', 'task_loss', 'spike_loss', 'entropy_loss', 'spiking']}

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

			# Display network performance
			if i%20 == 0:
				spiking = (par['num_time_steps']/par['dt'])*np.mean(spike)
				acc = get_perf(trial_info['desired_output'], output, trial_info['train_mask'])

				data_record['iter'].append(i)
				data_record['acc'].append(acc)
				data_record['task_loss'].append(task_loss)
				data_record['spiking'].append(spiking)

				trials = 4
				fig, ax = plt.subplots(5,trials, figsize=[12,8])
				for b in range(trials):
					ax[0,b].imshow(trial_info['neural_input'][:,b,:].T, aspect='auto')
					ax[1,b].imshow(trial_info['desired_output'][:,b,:].T, aspect='auto')
					ax[2,b].imshow(output[:,b,:].T, aspect='auto')
					ax[3,b].imshow(state[:,b,:].T, aspect='auto')
					ax[4,b].imshow(spike[:,b,:].T, aspect='auto')

				ax[0,0].set_ylabel('Network input')
				ax[1,0].set_ylabel('Expected Output')
				ax[2,0].set_ylabel('Network Output')
				ax[3,0].set_ylabel('Membrane Voltage')
				ax[4,0].set_ylabel('Spike Output')
				ax[4,0].set_xlabel('Time')

				plt.savefig('./savedir/iter{}_outputs.png'.format(i))
				plt.clf()
				plt.close()

				pickle.dump(data_record, open(par['savedir'] + par['save_fn'] + '.pkl', 'wb'))

				print('Iter: {:>6} | Accuracy: {:5.3f} | Task Loss: {:5.3f} | Spike Rate: {:6.2f} Hz'.format(\
						i, acc, task_loss, spiking))



		if par['save_analysis']:
			save_results = {'task':task, 'accuracy_curve':accuracy_curve, 'loss_curve':loss_curve, 'par':par}
			pickle.dump(save_results, open(par['savedir'] + save_fn + '.pkl', 'wb'))

	print('\nModel execution complete.')


def get_perf(target, output, mask):
	""" Calculate task accuracy by comparing the actual network output
	to the desired output only examine time points when test stimulus is
	on in another words, when target[:,:,-1] is not 0 """

	output = np.stack(output, axis=0)
	mk = mask*np.reshape(target[:,:,0] == 0, (par['num_time_steps'], par['batch_size']))

	target = np.argmax(target, axis = 2)
	output = np.argmax(output, axis = 2)

	return np.sum(np.float32(target == output)*np.squeeze(mk))/np.sum(mk)


if __name__ == '__main__':
	try:
		if len(sys.argv) > 1:
			main(sys.argv[1])
		else:
			main()
	except KeyboardInterrupt:
		quit('Quit by KeyboardInterrupt.')
