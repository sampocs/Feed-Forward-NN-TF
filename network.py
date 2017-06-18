import tensorflow as tf 

#MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sesh = tf.InteractiveSession()

#Config
learning_rate = 0.5
num_steps = 10000

def main ():
	#Input/Output
	x = tf.placeholder(tf.float32, shape=[None,784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
		#None means the batch can be any size

	#Weights and bias
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	sesh.run(tf.global_variables_initializer())

	#Feed forward
	y = tf.matmul(x, W) + b

	#Loss (softmax -> cross entropy loss -> average), SGD
	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	#Train
	for _ in range(10000):
		batch = mnist.train.next_batch(100)
		train_step.run(feed_dict={x: batch[0], y_: batch[1]}) #Able to run since session is interactive
			#feed_dict can replace any tensor

	#Test
	#Get list of booleans for each instance (True if prediction is correct)
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	print "test accuracy: " + str(accuracy.eval(feed_dict={x: mnist.test.images, 
									y_: mnist.test.labels}))

if __name__ == "__main__":
	main()

