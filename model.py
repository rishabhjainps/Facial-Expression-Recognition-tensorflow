import tensorflow as tf 
import numpy as np

FLAGS = tf.app.flags.FLAGS

# MAIN CNN MODEL
class Model:
	"""Define structure of Main Model building tensorflow graph

		Variables: 
			outputs : Contains all outputs from all the layers from input to output of model

	"""

	def __init__(self, name, features ):
		"""Constructor for Model 
		
			Args:
				name    : Specifies name of Model
				features: Input features to model of Batch_Size*48*48*1 shape 
		"""
		self.name = name
		self.outputs = [ features ]
		
	def get_last_output(self):
		"""Returns the last ouput inserted in outputs list while building the graph	 
			Used for input to the next layer
		"""
		return self.outputs[-1]

	def get_layer_str(self, layer=None):
		"""Returns the layer number in string giving unique string to every layer
			If layer not specified return last layer unique string
		"""
		if layer is None:
			layer = self.get_num_layers()
		
		return '%s_L%03d' % (self.name, layer+1)

	def get_num_layers(self):
		"""Returns the number of layers in the model till now 
		"""
		return len(self.outputs)

	def get_num_inputs(self):
		"""Return the last layer inserted output channels [*,*,channels] or new layer input shape returning number of 
			channels of last layer 
		"""
		return self.get_last_output().get_shape()[-1]

	def add_conv2d(self, kernel_size=1, output_channels=32, stride=1, stddev_factor=0.1):
		"""Adds convolution layer to the last in the model
		"""
		# Previous layer should be 4-dimension ( batch, width, height, channels )
		assert len(self.get_last_output().get_shape()) == 4

		with tf.variable_scope( self.get_layer_str() ):
			input_channels = self.get_num_inputs()

			# Initialize weight variable with shape [kernel_size,kernel_size,input_channels,output_channels]
			# between previous layer and new layer 
			weight = tf.get_variable('weight',shape=[kernel_size,kernel_size,input_channels,output_channels],
				initializer=tf.contrib.layers.xavier_initializer())

			# Compute the weight decay term
			#  decay term (lambd)*||w||^2 
			weight_decay = tf.multiply( tf.nn.l2_loss(weight) , FLAGS.lambd )

			# Add all (lambd)*||w||^2 in this layer to the losses
			tf.add_to_collection('losses',weight_decay)

			# Define the convolution layer with output of last layer as input to this layer and define kernel 
			# out will be of 4-dimension ( [ batch, ceil(previous_layer_height/stride) , ceil(previous_layer_width/stride) ,output_channels ] )
			output = tf.nn.conv2d(self.get_last_output(), weight, strides=[1,stride,stride,1], padding='SAME' )

			# Initialize b as zeros of shape [output_channels]
			initb = tf.constant(0.0 , shape=[output_channels])
			bias = tf.get_variable('bias' , initializer=initb )

			# Add Summaries to tensorflow board
			tf.summary.histogram("weight",weight)
			tf.summary.histogram("bias"  ,bias)

			# Add bias to the output 
			output = tf.nn.bias_add( output,bias )

		self.outputs.append(output)
		return self

	def add_max_pool(self, ksize, strides, padding="SAME"):
		"""Add MaxPool to the model, input shape is same as output shape
		"""
		with tf.variable_scope(self.get_layer_str()):
			output = tf.nn.max_pool(self.get_last_output(),ksize=[1,ksize,ksize,1],strides=[1,strides,strides,1],padding=padding, data_format='NHWC')

		self.outputs.append(output)
		return self

	def add_batch_norm(self , is_training):
		"""Add batch norm to the model, input shape is same as output shape
		"""
		with tf.variable_scope(self.get_layer_str()):
			output =tf.contrib.layers.batch_norm(self.get_last_output(),center=True, scale=True,updates_collections=None,  is_training=is_training,scope='bn')
		
		self.outputs.append(output)
		return self

	def add_relu(self):
		"""Add Linear Rectifier to the model, input shape is same as output shape
		"""
		with tf.variable_scope(self.get_layer_str()):
			output = tf.nn.relu(self.get_last_output())

		self.outputs.append(output)
		return self

	def add_dropout(self, keep_prob, is_training):
		"""Add dropout to the model, dropout keep_prob decides number of neurons to be closed
		   With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, otherwise outputs 0.
		   NO DROPOUT SHOULD BE USED WHILE TESTING OR CALCULATING RESULTS
		"""
		with tf.variable_scope(self.get_layer_str()):
			output = tf.cond( is_training, 
				lambda: tf.nn.dropout( self.get_last_output(), keep_prob ) , 
				lambda: tf.nn.dropout( self.get_last_output(), 1.0 ) )

		self.outputs.append(output)
		return self

	def flatten(self):
		"""Flattens the last layer output while maintaining the batch_size.
		   Assumes that the first dimension represents the batch.
		"""
		with tf.variable_scope(self.get_layer_str()):
			output = tf.contrib.layers.flatten(self.get_last_output())

		self.outputs.append(output)
		return self

	def add_fully_connected_with_relu(self , num_output ):
		"""Adds a fully connected layer with relu activation
		"""
		with tf.variable_scope(self.get_layer_str()):
			output = tf.contrib.layers.fully_connected(self.get_last_output(), num_output)

		self.outputs.append(output)
		return self	

	def add_fully_connected(self , num_output ):
		"""Adds a fully connected layer
		   Useful in last layer without relu
		"""
		with tf.variable_scope(self.get_layer_str()):
			output = tf.contrib.layers.fully_connected(self.get_last_output(), num_output ,activation_fn=None)

		self.outputs.append(output)
		return self


def convolutional_nn(sess, features, labels, is_training):
	
	old_vars = tf.global_variables()

	model = Model('CNN', features)

	# define layer_list as tuples of layers where each tuple has values (kernel_size,output_channels)
	cnn_layer_list = [(3,64),(5,128)]

	for layer in cnn_layer_list :
		model.add_conv2d(layer[0],layer[1])
		model.add_batch_norm(is_training)
		model.add_relu()
		model.add_max_pool(2,2)
		model.add_dropout(0.25,is_training)

	model.flatten()
	# define fully connected as tuples defining number of outputs
	fc_layer_list = [256]

	# Fully connected layers
	for layer in fc_layer_list:
		model.add_fully_connected(layer)
		model.add_batch_norm(is_training)
		model.add_relu()
		model.add_dropout(0.25,is_training)

	# Output layer
	num_outputs = 7
	model.add_fully_connected(num_outputs)

	# collect all variables for optimization
	new_vars = tf.global_variables()
	cnn_vars = list(set(new_vars)-set(old_vars))

	return model.get_last_output() , cnn_vars 

def create_model(sess, features, labels):
	""" Create the deep cnn model
	"""
	height   = int( features.get_shape()[1] )
	width    = int( features.get_shape()[2] )
	channels = int( features.get_shape()[3] )

	is_training = tf.placeholder(tf.bool)

	with tf.variable_scope('cnn') as scope:
		output, cnn_vars = convolutional_nn(sess, features, labels,is_training)
		scope.reuse_variables()

	return output, cnn_vars, is_training

def compute_loss(cnn_output, labels):
	"""Calculate total loss with regulariztion loss
	"""
	softmax_loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=cnn_output) )
	tf.add_to_collection('losses',softmax_loss)

	total_loss = tf.add_n( tf.get_collection('losses'), name='total_loss' )

	return total_loss, softmax_loss

def compute_accuracy(sess):
	"""Calculates accuracy of eval_input running graph on basis eval_label provided as placeholder
		provided using feed_dict. 
		Used for calculating training and dev accuracy 
	"""
	eval_input = tf.placeholder(tf.float32, shape=[FLAGS.BATCH_SIZE,48,48,1])
	eval_label = tf.placeholder(tf.float32, shape=[FLAGS.BATCH_SIZE])

	eval_label2 = tf.cast(eval_label,tf.int64)
	is_training = tf.placeholder(tf.bool)

	with tf.variable_scope('cnn') as scope:
		scope.reuse_variables()
		output, _ =  convolutional_nn(sess, eval_input, eval_label,is_training)

	accuracy = tf.contrib.metrics.accuracy(tf.argmax(output, 1), eval_label2)

	return output, eval_input, eval_label, accuracy , is_training

def create_optimizer(total_loss, var_list):
	"""Create an optimizer with desired parameter
	"""
	# global_step refer to the number of batches seen by the graph.
	global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
	learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer')

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		minimize  = optimizer.minimize(total_loss, var_list=var_list, name='loss_minimize', global_step=global_step )

	return (global_step,learning_rate,minimize)