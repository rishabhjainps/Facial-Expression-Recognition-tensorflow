import tensorflow as tf 

FLAGS = tf.app.flags.FLAGS

# Load input in queue input stream
# Used to receive following data in batches of specified size as defined by FLAGS.BATCH_SIZE
def get_files(filename):
	total_input_queue = tf.train.string_input_producer([filename],shuffle=False)
	
	feature_reader = tf.TextLineReader()
	
	# read from tf.TextLineReader()
	_,value = feature_reader.read(total_input_queue)

	record_defaults = [ [0] , ["0 0"] , ["Public"] ]
	label,image,_ = tf.decode_csv(value,record_defaults=record_defaults,field_delim=",")
	
	# The output of string_split is not a tensor, instead, it is a SparseTensorValue. Therefore, it has a property value that stores the actual values. as a tensor. 
	image_splitted = tf.string_split([image], delimiter=' ')
	image_values = tf.reshape(image_splitted.values, [48,48,1])
	
	# string_to_number will convert the feature's numbers into float32 as I need them. 
	image_numbers = tf.string_to_number(image_values, out_type=tf.float32)

	image_numbers = tf.divide( image_numbers , 255.0 )

	# Get the batches from the defined images
	image_batch , label_batch = tf.train.batch([image_numbers,label],batch_size=FLAGS.BATCH_SIZE) 

	return image_batch , label_batch 


#TESTING WITH AN EXAMPLE
#Don't use with fer2013.csv
'''
testDir = "./data/ferTest.csv"

with tf.Session() as sess:
  # Start populating the filename queue.
  features , labels = get_files(sess,testDir)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess,coord=coord)

  for i in range(1):

	# Retrieve in batches:
	example, label = sess.run([features, labels])
	print(example)
	print(label)
'''