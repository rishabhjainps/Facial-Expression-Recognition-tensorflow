import tensorflow as tf 
import numpy as numpy

import input_pipeline
import model

FLAGS = tf.app.flags.FLAGS

evaluate_dir = './data/ferTest.csv'

# basic parameters for easy setup
EVALUATE_DATASET_SIZE = 3589

tf.app.flags.DEFINE_integer('BATCH_SIZE', 64, '')
tf.app.flags.DEFINE_float('lambd',0.01 ,'Lambda value for regularization')

def setup_tensorflow():
	"""Basic Config for starting a session
	"""
	config = tf.ConfigProto(log_device_placement=False)
	sess = tf.Session(config=config)
	
	with sess.graph.as_default():
		tf.set_random_seed(0)

	random.seed(0)
	return sess

def evaluate_model():
	"""Evaluate model with calculating test accuracy
	"""
	sess = setup_tensorflow()

	# SetUp Input PipeLine for queue inputs
	with tf.name_scope('train_input'):
		evaluate_features, evaluate_labels = input_pipeline.get_files(evaluate_dir)

	# Create Model creating graph
	output, var_list, is_training1 = model.create_model(sess, evaluate_features, evaluate_labels)

	# Create Model loss  & optimizer
	with tf.name_scope("loss"):
		total_loss, softmax_loss  = model.compute_loss(output, evaluate_labels)

	(global_step, learning_rate, minimize) = model.create_optimizer(total_loss, var_list)	

	# Acurracy setup 
	out_eval,eval_input, eval_label, accuracy, is_training2 = model.compute_accuracy(sess)

	sess.run(tf.global_variables_initializer())
	
	# Basic stuff for input pipeline
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess,coord=coord)

	# Calculate number of batches to run
	num_batches = EVALUATE_DATASET_SIZE/FLAGS.BATCH_SIZE

	# Add ops to restore all the variables.
	saver = tf.train.Saver()

	# Give the path of model with weights u wanna load
	saver.restore(sess, "./model/model100.ckpt")

	# Calculate acurracy for whole evaluate data
	total_accuracy = 0
	
	for batch in range(1,num_batches+1 ):

		# Load input from the pipeline in batches , batch by batch
		input_batch, label_batch = sess.run([evaluate_features, evaluate_labels])

		feed_dict = {eval_input:input_batch,eval_label:label_batch,is_training2:False}
		ops = [out_eval,accuracy]

		# Get the accuracy on evaluate batch run
		_,acc = sess.run(ops, feed_dict=feed_dict)

		print(" batch /" + str (batch) + " /" + str(num_batches) + " acc: " + str( acc ) )
		total_accuracy += acc
	
	total_accuracy /= (num_batches+1)

	# Total Accuracy for Evaluate dataset
	print(" ACCURACY : " + str( total_accuracy ) )


evaluate_model()
