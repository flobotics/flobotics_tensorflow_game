import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import tensorflow as tf
import Tkinter
from PIL import Image, ImageTk
import os.path
import time


max_degree = 200 
degree_goal = 30 
current_degree = 10 


STATE_FRAMES = 4
NUM_ACTIONS = 3  #stop,left,right 
NUM_STATES = 400 
MEMORY_SIZE = 3000
OBSERVATION_STEPS = 500
MINI_BATCH_SIZE = 30

RESIZED_DATA_X = 20  #NUM_STATES resized to 20x20
RESIZED_DATA_Y = 20 
FUTURE_REWARD_DISCOUNT = 0.9


probability_of_random_action = 1.01 
sum_writer_index = 0
train_play_loop = 10

data = None
photo = None
root = None
canvas = None
not_random = 1
random_loop = 0
delayer = 0

#build a 20x20 array with two pixels with value 1, all other value 0
def get_current_state():
	global current_degree
	global degree_goal

	a = np.zeros([RESIZED_DATA_X*10])
	a[current_degree] = 1
	b = np.zeros([RESIZED_DATA_X*10])
	b[degree_goal] = 1
	c = []
	c.extend(a)
	c.extend(b)
	c = np.reshape(c, (20,20))
	return c

#if we are in the same position as the second array, we get reward
#we reshape into two arrays, first array is the pixel which can be moved, the second array is the goal
def get_reward(current_state):
	s = np.reshape(current_state, (2, RESIZED_DATA_X*10))
	r = s[0] * s[1]
	r = sum(r)
	return r

#we choose a random or learned action
def choose_next_action(last_state):
	new_action = np.zeros([NUM_ACTIONS])
	global probability_of_random_action
	global random_loop
	global not_random

	#simple decreaseing
	if not_random == 1:
		random_loop += 1
		if delayer == 1:
			if random_loop >= 100:
				probability_of_random_action -= 0.00001
				random_loop = 0

	#print probability_of_random_action
	
	if random.random() < probability_of_random_action:
		new_action_index = random.randint(0,2)
		new_action[new_action_index] = 1
		#print new_action
	else:
		readout_t = session.run(output_layer, feed_dict={input_layer: [last_state]})
		r1 = np.asarray(readout_t)
		r1 = np.reshape(r1, (NUM_ACTIONS))
		action_index = np.argmax(readout_t)
		new_action[action_index] = 1
		#print new_action
	
	return new_action

def weight_variable(shape, name):
    	initial = tf.truncated_normal(shape, stddev=0.01)
    	return tf.Variable(initial)

def bias_variable(shape, name):
    	initial = tf.constant(0.1, shape=shape)
    	return tf.Variable(initial)

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# here we do the action, which means, change the game environment (state)
# we can only stop, go one pixel right or left
def do_action(action):
	global current_degree

	if action[0] == 1:
		current_degree = current_degree
	if action[1] == 1:
		current_degree += 1
	if action[2] == 1:
		current_degree -= 1

	if current_degree > 50: #199: #263:
        	current_degree = 50#199 #263
        elif current_degree < 0:
        	current_degree = 0

	

def train(observations):
	#print("train")
	global sum_writer_index

	mini_batch = random.sample(observations, MINI_BATCH_SIZE)
	previous_states = [d[0] for d in mini_batch]
        actions = [d[1] for d in mini_batch]
        rewards = [d[2] for d in mini_batch]
        current_states = [d[3] for d in mini_batch]

	agents_expected_reward = []

        agents_reward_per_action = session.run(output_layer, feed_dict={input_layer: current_states})

        for i in range(len(mini_batch)):
        	agents_expected_reward.append(rewards[i] + FUTURE_REWARD_DISCOUNT * np.max(agents_reward_per_action[i]))

	_, result = session.run([train_operation, merged], feed_dict={input_layer: previous_states, action : actions, target: agents_expected_reward, keep_prob: 0.5})
	sum_writer.add_summary(result, sum_writer_index)
	sum_writer_index += 1

#Tkinter loop
def image_loop():
        global data
        global photo
	global canvas
	global root

        im=Image.fromstring('L', (data.shape[1],data.shape[0]), data.astype('b').tostring())
        photo = ImageTk.PhotoImage(master = canvas, image=im)
        canvas.create_image(10,10,image=photo,anchor=Tkinter.NW)
        root.update()

        root.after(100,image_loop)



######  main python program starts here  #####

observations = deque()
first_time = 1
last_state = None


#######TK inter 
root = Tkinter.Tk()
frame = Tkinter.Frame(root, width=75, height=75)
frame.pack()
canvas = Tkinter.Canvas(frame, width=75,height=75)
canvas.place(x=-2,y=-2)
root.after(1000,image_loop) # INCREASE THE 0 TO SLOW IT DOWN


#################### create network

session = tf.Session()

state = tf.placeholder("float", [None, NUM_STATES])
action = tf.placeholder("float", [None, NUM_ACTIONS])
target = tf.placeholder("float", [None])
keep_prob = tf.placeholder("float")

input_layer = tf.placeholder("float", [None, RESIZED_DATA_X, RESIZED_DATA_Y, STATE_FRAMES])

with tf.name_scope("conv1") as conv1:
	conv_weights_1 = weight_variable([1,1,4,32], "conv1_weights")
        conv_biases_1 = bias_variable([32], "conv1_biases")
        cw1_hist = tf.histogram_summary("conv1/weights", conv_weights_1)
        cb1_hist = tf.histogram_summary("conv1/biases", conv_biases_1)
        c1 = tf.reshape(conv_weights_1, [32, 1, 1, 4])
        cw1_image_hist = tf.image_summary("conv1_w", c1)
	
	l2n_input_layer = tf.nn.l2_normalize(input_layer, 0)
	
	h_conv1 = tf.nn.relu(tf.nn.conv2d(l2n_input_layer, conv_weights_1, strides=[1, 1, 1, 1], padding="VALID") + conv_biases_1)
        
	bn_conv1_mean, bn_conv1_variance = tf.nn.moments(h_conv1,[0,1,2,3])
        bn_conv1_scale = tf.Variable(tf.ones([32]))
        bn_conv1_offset = tf.Variable(tf.zeros([32]))
        bn_conv1_epsilon = 1e-3
	bn_conv1 = tf.nn.batch_normalization(h_conv1, bn_conv1_mean, bn_conv1_variance, bn_conv1_offset, bn_conv1_scale, bn_conv1_epsilon)
	
	h_pool1 = max_pool_2x2(bn_conv1)
	#h_pool1 = tf.nn.max_pool(bn_conv1, ksize=[1,1,1,1], strides=[1,1,1,1], padding='SAME')
	

with tf.name_scope("conv2") as conv2:
        conv_weights_2 = weight_variable([2,2,32,64], "conv2_weights")
        conv_biases_2 = bias_variable([64], "conv2_biases")
        cw2_hist = tf.histogram_summary("conv2/weights", conv_weights_2)
        cb2_hist = tf.histogram_summary("conv2/biases", conv_biases_2)
        c2 = tf.reshape(conv_weights_2, [32,64,1,4])
        cw2_image_hist = tf.image_summary("conv2_w", c2)

	h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, conv_weights_2, strides=[1, 2, 2, 1], padding="SAME") + conv_biases_2)

	bn_conv2_mean, bn_conv2_variance = tf.nn.moments(h_conv2, [0,1,2,3])
	bn_conv2_scale = tf.Variable(tf.ones([64]))
	bn_conv2_offset = tf.Variable(tf.zeros([64]))
	bn_conv2_epsilon = 1e-3
	bn_conv2 = tf.nn.batch_normalization(h_conv2, bn_conv2_mean, bn_conv2_variance, bn_conv2_offset, bn_conv2_scale, bn_conv2_epsilon)

	#h_pool2 = max_pool_2x2(h_conv2)
	h_pool2 = tf.nn.max_pool(bn_conv2, ksize=[1,1,1,1], strides=[1,1,1,1], padding='SAME')
#
#
#with tf.name_scope("conv3") as conv3:
#        conv_weights_3 = weight_variable([1,1,64,64], "conv3_weights")
#        conv_biases_3 = bias_variable([64], "conv3_biases")
#        cw3_hist = tf.histogram_summary("conv3/weights", conv_weights_3)
#        cb3_hist = tf.histogram_summary("conv3/biases", conv_biases_3)
#        c3 = tf.reshape(conv_weights_3, [64,64,1,1])
#        cw3_image_hist = tf.image_summary("conv3_w", c3)
#
#	h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, conv_weights_3, strides=[1,1,1,1], padding="SAME") + conv_biases_3)
#	bn_conv3_mean, bn_conv3_variance = tf.nn.moments(h_conv3, [0,1,2,3])
#	bn_conv3_scale = tf.Variable(tf.ones([64]))
#	bn_conv3_offset = tf.Variable(tf.zeros([64]))
#	bn_conv3_epsilon = 1e-3
#
#	h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1,1,1,1], strides=[1,1,1,1], padding='SAME')


with tf.name_scope("fc_1") as fc_1:
        fc1_weights = weight_variable([5*5*64, 512], "fc1_weights")
        fc1_biases = bias_variable([512], "fc1_biases")
        fc1_b_hist = tf.histogram_summary("fc_1/biases", fc1_biases)
        fc1_w_hist = tf.histogram_summary("fc_1/weights", fc1_weights)

	h_pool3_flat = tf.reshape(h_pool2, [-1,5*5*64])

	#h_pool3_flat = tf.reshape(h_pool1, [-1,128])
	#h_pool3_flat = tf.reshape(h_pool2, [-1,5*5*64])
	final_hidden_activation = tf.nn.relu(tf.matmul(h_pool3_flat, fc1_weights, name='final_hidden_activation') + fc1_biases)
	#final_hidden_activation = tf.nn.relu(tf.matmul(h_fc1_drop, fc1_weights, name='final_hidden_activation') + fc1_biases)




with tf.name_scope("fc_2") as fc_2:
        fc2_weights = weight_variable([512, NUM_ACTIONS], "fc2_weights")
        fc2_biases = bias_variable([NUM_ACTIONS], "fc2_biases")
        fc2_w_hist = tf.histogram_summary("fc_2/weights", fc2_weights)
        fc2_b_hist = tf.histogram_summary("fc_2/biases", fc2_biases)

	#keep_prob = tf.placeholder(tf.float32)

        h_fc1_drop = tf.nn.dropout(final_hidden_activation, 0.5 )


	#output_layer = tf.matmul(final_hidden_activation, fc2_weights) + fc2_biases
	output_layer = tf.matmul(h_fc1_drop, fc2_weights) + fc2_biases
	ol_hist = tf.histogram_summary("output_layer", output_layer)




#we feed in the action the NN would do and targets=rewards ???
with tf.name_scope("readout"):
	readout_action = tf.reduce_sum(tf.mul(output_layer, action), reduction_indices=1)
	r_hist = tf.histogram_summary("readout_action", readout_action)

with tf.name_scope("loss_summary"):
	#loss = tf.reduce_mean(tf.square(output - target))
	loss = tf.reduce_mean(tf.square(target - readout_action))
        #loss = tf.reduce_mean(tf.square(output_layer - target))
        tf.scalar_summary("loss", loss)

merged = tf.merge_all_summaries()

sum_writer = tf.train.SummaryWriter('/tmp/train/c/', session.graph)

train_operation = tf.train.AdamOptimizer(0.0001, epsilon=0.000001).minimize(loss)


session.run(tf.initialize_all_variables())
saver = tf.train.Saver()

if os.path.isfile("/home/ros/tensorflow-models/model-mini.ckpt"):
	saver.restore(session, "/home/ros/tensorflow-models/model-mini.ckpt")
	print "model restored"


########### end create network


data=np.array(np.random.random((RESIZED_DATA_X, RESIZED_DATA_Y))*100,dtype=int)
obs = 0
obs_s = 0

try:
	while True:
		#tkinter update
		root.update_idletasks()
		root.update()

		#print("test if it loops or blocks")
		state_from_env = get_current_state()
		reward = get_reward(state_from_env)
		#time.sleep(0.1)
		
		##tkinter update
		global data
		data=state_from_env
		data = data * 255 #the value 1 * 255=255 => white pixel to see

		state_from_env = (( (state_from_env * 255) - 128) / 128)
	
		#if we run for the first time, we build a state
		if first_time == 1:
			first_time = 0
			last_state = np.stack(tuple(state_from_env for _ in range(STATE_FRAMES)), axis=2)
			last_action = np.zeros([NUM_ACTIONS])  #speeed of both servos 0
			#do_action() which does nothing


		state_from_env = state_from_env.reshape(RESIZED_DATA_X, RESIZED_DATA_Y, 1)
		#state_from_env = np.linalg.norm(state_from_env)
		current_state = np.append(last_state[:,:,1:], state_from_env, axis=2)

		observations.append((last_state, last_action, reward, current_state))	

		if len(observations) > MEMORY_SIZE:
			#print("POPLEFT---------------------")
			observations.popleft()

		#print len(observations)
		#if len(observations) % OBSERVATION_STEPS == 0:
		obs += 1
		obs_s += 1
		if obs > OBSERVATION_STEPS:
			obs = 0
			#for i in range(OBSERVATION_STEPS/MINI_BATCH_SIZE):
			train(observations)
		
			#print "save model"
			if obs_s > 1000:
			        save_path = saver.save(session, "/home/ros/tensorflow-models/model-mini.ckpt")
				obs_s = 0

		last_state = current_state
		last_action = choose_next_action(last_state)

		#if we got the max reward, we change degree_goal mostly, perhaps sometimes force_1/2_goal
		if reward == 1:
			print("MAX REWARD -------- NEW DEGREE GOAL")
			global train_play_loop
			global probability_of_random_action
			global not_random

			print probability_of_random_action
			print train_play_loop
			
			#degree_goal = random.randint(0, (max_degree-1) )
			degree_goal = random.randint(0, (50-1) )

			if train_play_loop <= 0:
				delayer = 1
				t = raw_input("train or play? input 0 for play, number for how often it train and find degree_goal: ")
				t = int(t)
				if t == 0:
					nb = raw_input("new degree_goal: ")
					degree_goal = int(nb)
					print degree_goal
					nb = raw_input("new probability of choosing random action: 0.0-1.0 : ")
					probability_of_random_action = float(nb)
					print probability_of_random_action
					nr = raw_input("random or not 0 or 1 : ")
					not_random = int(nr)
					print not_random
					train_play_loop = 1 #just to decrease it some steps later to 0 and not a negative number
				else:
					train_play_loop = t
					
			train_play_loop -= 1

		do_action(last_action)
	
except KeyboardInterrupt:
	print "save model"
	save_path = saver.save(session, "/home/ros/tensorflow-models/model-mini.ckpt")
	session.close()



