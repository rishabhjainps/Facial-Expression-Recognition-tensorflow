import tensorflow as tf 
import numpy as numpy

import os 
import glob
import random

import input_pipeline
import model

FLAGS = tf.app.flags.FLAGS

train_dir = './data/ferTrain.csv'
dev_dir = './data/ferDev.csv'

output_dir = './output/'
model_dir = './model/'
summary_dir = './logs/'

# basic parameters for easy setup
TRAINING_DATASET_SIZE = 28709
DEV_DATASET_SIZE = 3589

# training parameters
EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Logging Parameters
SUMMARY_PERIOD=10
CHECKPOINT_PERIOD=10

tf.app.flags.DEFINE_integer('TRAINING_SIZE', 28709 , '')
tf.app.flags.DEFINE_integer('BATCH_SIZE', 64, '')
tf.app.flags.DEFINE_float('lambd',0.001 ,'Lambda value for regularization')



def setup_tensorflow():
    config = tf.ConfigProto(log_device_placement=False)
    sess = tf.Session(config=config)
    
    with sess.graph.as_default():
        tf.set_random_seed(0)

    random.seed(0)
    return sess

def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def train_model():

	sess = setup_tensorflow()

	with tf.name_scope('train_input'):
		train_features, train_labels = input_pipeline.get_files(train_dir)
	with tf.name_scope('dev_input'):
		dev_features , dev_labels  = input_pipeline.get_files(dev_dir)

	output, var_list = model.create_model(sess, train_features, train_labels)

	with tf.name_scope("loss"):
		total_loss, softmax_loss  = model.compute_loss(output, train_labels )
		tf.summary.scalar("loss",total_loss)

	(global_step, learning_rate, minimize) = model.create_optimizer(total_loss, var_list)	

	tf.summary.scalar("loss",total_loss)

	#acurracy setup
	out_eval,eval_input, eval_label, accuracy = model.compute_accuracy(sess)


	sess.run(tf.global_variables_initializer())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess,coord=coord)

	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()


	num_batches = TRAINING_DATASET_SIZE/FLAGS.BATCH_SIZE
	num_batches_dev = DEV_DATASET_SIZE/FLAGS.BATCH_SIZE

	#add computation graph to summary writer
	writer = tf.summary.FileWriter(summary_dir)
	writer.add_graph(sess.graph)

	merged_summaries = tf.summary.merge_all()

	for epoch in range(1,EPOCHS+1):


		Tsloss = 0
		Tloss = 0

		for batch in range(1,num_batches+1 ):
			feed_dict = {learning_rate: LEARNING_RATE}
			ops = [minimize, softmax_loss, total_loss, merged_summaries]
			_, sloss, loss, summaries = sess.run(ops, feed_dict=feed_dict)
			#print ("Epoch /" + str (epoch) + " /" + str(EPOCHS)+" batch /" + str (batch) + " /" + str(num_batches)   + " ; Loss " + str(Tloss)+ " softmax Loss " + str(Tsloss))
			Tsloss += sloss
			Tloss  += loss

		Tsloss /= (num_batches+1)
		Tloss /= (num_batches+1)

		print ("Epoch /" + str (epoch) + " /" + str(EPOCHS)  + " ; Loss " + str(Tloss)+ " softmax Loss " + str(Tsloss))

		# calculate training acurracy
		total_accuracy = 0

		for batch in range(1,num_batches+1 ):

			input_batch, label_batch = sess.run([train_features, train_labels])

			feed_dict = {eval_input:input_batch,eval_label:label_batch}
			ops = [out_eval,accuracy]
			_,acc = sess.run(ops, feed_dict=feed_dict)

			#print(" TRAINING ACCURACY : " + str( acc ) )
			total_accuracy += acc
		
		total_accuracy /= (num_batches+1)

		print(" TRAINING ACCURACY : " + str( total_accuracy ) )


		# calculate dev acurracy

		# calculate training acurracy
		total_accuracy = 0

		for batch in range(1,num_batches_dev+1 ):

			input_batch, label_batch = sess.run([dev_features, dev_labels])

			feed_dict = {eval_input:input_batch,eval_label:label_batch}
			ops = [out_eval,accuracy]
			_,acc = sess.run(ops, feed_dict=feed_dict)

			total_accuracy += acc
		
		total_accuracy /= (num_batches_dev+1)

		print(" DEV ACCURACY : " + str( total_accuracy ) )
		# write summary to logdir
		writer.add_summary(summaries)
		print "Summary Written to Logdir"

		make_dir_if_not_exists(model_dir)
		save_path = saver.save(sess, model_dir + "model" + str(epoch) +".ckpt")
		print("Model saved in path: %s" % save_path)
            
train_model()
