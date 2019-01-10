### Authors: Nicolas Y. Masse, Gregory D. Grant

import numpy as np

print('\n--> Loading parameters...')

##############################
### Independent parameters ###
##############################

par = {
	# Setup parameters
	'savedir'				: './savedir/',
	'save_fn'				: 'spiking_testing',
	'training_method'		: 'SL',

	# Cell configuration
	'cell_type'				: 'adex',	# 'rate', 'lif', 'adex'
	'exc_model'				: 'RS',
	'inh_model'				: 'cNA',
	'current_divider'		: 3e8,
	'input_frequency'		: 30,		# Matches Hz with tuning_height of 4.0 and kappa of 2.0
	'spiking_target'		: 2,
	'spiking_cost'			: 1e-3,
	'entropy_cost'			: 1e-5,

	# Network configuration
	'use_stp'				: True,
	'exc_inh_prop'			: 0.8,		# Literature 0.8, for EI off 1

	# Network shape
	'num_motion_tuned'		: 64,
	'num_fix_tuned'			: 0,
	'num_rule_tuned'		: 0,
	'num_receptive_fields'	: 1,
	'n_hidden'				: 400,
	'n_output'				: 3,
	'n_val'					: 1,
	'include_rule_signal'	: True,

	# Timings and rates
	'dt'					: 1,
	'learning_rate'			: 1e-3,
	'membrane_time_constant': 100,
	'connection_prob'		: 1.0,

	# Variance values
	'clip_max_grad_val'     : 1.0,
	'input_mean'            : 0.0,
	'noise_in_sd'           : 0.0,
	'noise_rnn_sd'          : 0.05,

	# Task specs
	'task'                  : 'dms',  # See stimulus file for more options
	'kappa'                 : 2.0,
    'tuning_height'         : 4.0,
    'response_multiplier'   : 2.0,
    'num_rules'             : 1,

    # Task timings
    'dead_time'             : 20,
    'fix_time'              : 80,
    'sample_time'           : 200,
    'delay_time'            : 500,
    'test_time'             : 200,
    'mask_time'             : 20,

	# Tuning function data
	'num_motion_dirs'       : 8,

	# Synaptic plasticity specs
	'tau_fast'              : 100,
	'tau_slow'              : 1000,
	'U_stf'                 : 0.15,
	'U_std'                 : 0.45,

	# Training specs
	'batch_size'            : 256,
	'n_train_batches'       : 50000,

}

############################
### Dependent parameters ###
############################

def update_parameters(updates):
	""" Updates parameters based on a provided
		dictionary, then updates dependencies """

	for (key, val) in updates.items():
		par[key] = val
		print('Updating: {:<24} --> {}'.format(key, val))

	update_dependencies()


def update_dependencies():
	""" Updates all parameter dependencies """


	# Generate EI matrix
	par['EI'] = True if par['exc_inh_prop'] < 1 else False
	par['num_exc_units'] = int(np.round(par['n_hidden']*par['exc_inh_prop']))
	par['EI_list'] = np.ones(par['n_hidden'], dtype=np.float32)
	par['EI_list'][par['num_exc_units']:] *= -1. if par['EI'] else 1.
	par['EI_matrix'] = np.diag(par['EI_list'])

	# Establish input neurons
	par['n_input'] = par['num_motion_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned']

	# Specify time step in seconds and set neuron time constant
	par['dt_sec'] = par['dt']/1000
	par['alpha_neuron'] = np.float32(par['dt'])/par['membrane_time_constant']

	# Generate noise deviations
	par['noise_rnn'] = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
	par['noise_in'] = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd']

	# Set trial step length
	par['trial_length'] = par['dead_time'] + par['fix_time'] + +par['sample_time'] + par['delay_time'] + par['test_time']
	par['num_time_steps'] = par['trial_length']//par['dt']

	# Set up weights and biases
	c = 0.05
	par['W_in_init'] = c*np.float32(np.random.gamma(shape=0.2, scale=1.0, size=[par['n_input'], par['n_hidden']]))
	par['W_out_init'] = np.float32(np.random.uniform(-c, c, size=[par['n_hidden'], par['n_output']]))

	if par['EI']:
		par['W_rnn_init'] = c*np.float32(np.random.gamma(shape=0.1, scale=1.0, size=[par['n_hidden'], par['n_hidden']]))
		par['W_rnn_init'][:, par['num_exc_units']:] = c*np.float32(np.random.gamma(shape=0.2, scale=1.0, size=[par['n_hidden'], par['n_hidden']-par['num_exc_units']]))
		par['W_rnn_init'][par['num_exc_units']:, :] = c*np.float32(np.random.gamma(shape=0.2, scale=1.0, size=[par['n_hidden']-par['num_exc_units'], par['n_hidden']]))
		par['W_rnn_mask'] = np.ones([par['n_hidden'], par['n_hidden']], dtype=np.float32) - np.eye(par['n_hidden'])
		par['W_rnn_init'] *= par['W_rnn_mask']



	else:
		par['W_rnn_init'] = np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))
		par['W_rnn_mask'] = np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32)

	par['b_rnn_init'] = np.zeros((1,par['n_hidden']), dtype=np.float32)
	par['b_out_init'] = np.zeros((1,par['n_output']), dtype=np.float32)

	###
	### Setting up synaptic plasticity parameters
	###

	"""
	0 = static
	1 = facilitating
	2 = depressing
	"""

	if par['use_stp']:
		par['alpha_stf']  = np.ones((1, par['n_hidden']), dtype=np.float32)
		par['alpha_std']  = np.ones((1, par['n_hidden']), dtype=np.float32)
		par['U']          = np.ones((1, par['n_hidden']), dtype=np.float32)

		par['syn_x_init'] = np.zeros((1, par['n_hidden']), dtype=np.float32)
		par['syn_u_init'] = np.zeros((1, par['n_hidden']), dtype=np.float32)

		for i in range(0,par['n_hidden'],2):
			par['alpha_stf'][0,i] = par['dt']/par['tau_slow']
			par['alpha_std'][0,i] = par['dt']/par['tau_fast']
			par['U'][0,i] = 0.15
			par['syn_x_init'][0,i] = 1
			par['syn_u_init'][0,i] = par['U'][0,i]

			par['alpha_stf'][0,i+1] = par['dt']/par['tau_fast']
			par['alpha_std'][0,i+1] = par['dt']/par['tau_slow']
			par['U'][0,i+1] = 0.45
			par['syn_x_init'][0,i+1] = 1
			par['syn_u_init'][0,i+1] = par['U'][0,i+1]

		par['stp_mod'] = par['dt_sec'] if par['cell_type'] == 'rate' else 1.

	### Adaptive-Exponential spiking
	if par['cell_type'] == 'adex':

		# Note that voltages are in units of V, A, and secs
		par['cNA'] = {
			'C'   : 59e-12,     'g'   : 2.9e-9,     'E'   : -62e-3,
			'V_T' : -42e-3,     'D'   : 3e-3,       'a'   : 1.8e-9,
			'tau' : 16e-3,      'b'   : 61e-12,     'V_r' : -54e-3,
			'Vth' : 0e-3,       'dt'  : par['dt']/1000 }
		par['RS']  = {
			'C'   : 104e-12,    'g'   : 4.3e-9,     'E'   : -65e-3,
			'V_T' : -52e-3,     'D'   : 0.8e-3,     'a'   : -0.8e-9,
			'tau' : 88e-3,      'b'   : 65e-12,     'V_r' : -53e-3,
			'Vth' : 0e-3,       'dt'  : par['dt']/1000 }

		par['adex'] = {}
		for (k0, v_exc), (k1, v_inh) in zip(par[par['exc_model']].items(), par[par['inh_model']].items()):
			assert(k0 == k1)
			par_matrix = np.ones([par['batch_size'],par['n_hidden']], dtype=np.float32)
			par_matrix[...,:int(par['n_hidden']*par['exc_inh_prop'])] *= v_exc
			par_matrix[...,int(par['n_hidden']*par['exc_inh_prop']):] *= v_inh
			par['adex'][k0] = par_matrix

		par['w_init'] = par['adex']['b']
		par['adex']['current_divider'] = par['current_divider']

	elif par['cell_type'] == 'lif':

		par['lif'] = {}
		par['lif']['Vth'] = 1.
		par['lif']['V_r'] = 0.
		par['lif']['membrane_time_constant'] = 20
		par['lif']['alpha_neuron'] = par['dt']/par['lif']['membrane_time_constant']

		par['w_init'] = 0.


update_dependencies()
print('--> Parameters successfully loaded.\n')
