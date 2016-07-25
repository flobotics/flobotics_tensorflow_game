import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import tensorflow as tf
import Tkinter
from PIL import Image, ImageTk
import os.path
import time


degree_goal = 9 
current_degree = 7 


STATE_FRAMES = 4
NUM_ACTIONS = 3  #stop,left,right 
MEMORY_SIZE = 300000
OBSERVATION_STEPS = 1000
MINI_BATCH_SIZE = 1000

RESIZED_DATA_X = 10 
RESIZED_DATA_Y = 2 
FUTURE_REWARD_DISCOUNT = 0.9


probability_of_random_action = 1.0 
sum_writer_index = 0
train_play_loop = 10

data = None
photo = None
root = None
canvas = None
random_loop = 0

accuracy = 0
steps_done = 0
steps_needed = 0
step = 0

#build a 10x2 array 
def get_current_state():
	global current_degree
	global degree_goal

	a = np.ones([RESIZED_DATA_X])
	a[current_degree] = 255
	b = np.ones([RESIZED_DATA_X])
	b[degree_goal] = 255
	c = []
	c.extend(a)
	c.extend(b)
	c = np.reshape(c, (RESIZED_DATA_X, RESIZED_DATA_Y))
	return c

#if we are in the same position as the second array, we get reward
#we reshape into two arrays, first array is the pixel which can be moved, the second array is the goal
def get_reward(current_state):
	s = np.reshape(current_state, (2, RESIZED_DATA_X))
	s1 = np.zeros([RESIZED_DATA_X])
	idx1 = np.argmax(s[0])
	s1[idx1] = 1
	s2 = np.zeros([RESIZED_DATA_X])
	idx2 = np.argmax(s[1])
	s2[idx2] = 1

	r = s1 * s2
	r = sum(r)
	return r

#we choose a random or learned action
def choose_next_action(last_state):
	new_action = np.zeros([NUM_ACTIONS])
	global probability_of_random_action
	global random_loop
	global not_random

	#simple decreaseing
	random_loop +=1
	if random_loop >= 100:
		probability_of_random_action -= 0.0001
		print probability_of_random_action
		random_loop = 0

	
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
	global steps_done

	if action[0] == 1:
		current_degree = current_degree
		#print("stop-action")
	if action[1] == 1:
		current_degree += 1
		#print("plus-action")
	if action[2] == 1:
		current_degree -= 1
		#print("minus-action")

	if current_degree > (RESIZED_DATA_X - 1): 
        	current_degree = (RESIZED_DATA_X - 1)
        elif current_degree < 0:
        	current_degree = 0

	steps_done += 1
	#print("do step", steps_done)

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

	_, result = session.run([train_operation, merged], feed_dict={input_layer: previous_states, action : actions, target: agents_expected_reward})
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

action = tf.placeholder("float", [None, NUM_ACTIONS])
target = tf.placeholder("float", [None])

input_layer = tf.placeholder("float", [None, RESIZED_DATA_X, RESIZED_DATA_Y, STATE_FRAMES])

with tf.name_scope("conv1") as conv1:
	conv_weights_1 = weight_variable([2,2,4,32], "conv1_weights")
        conv_biases_1 = bias_variable([32], "conv1_biases")
        cw1_hist = tf.histogram_summary("conv1/weights", conv_weights_1)
        cb1_hist = tf.histogram_summary("conv1/biases", conv_biases_1)
        c1 = tf.reshape(conv_weights_1, [32, 2,2, 4])
        cw1_image_hist = tf.image_summary("conv1_w", c1)
	
	h_conv1 = tf.nn.relu(tf.nn.conv2d(input_layer, conv_weights_1, strides=[1, 1, 1, 1], padding="SAME") + conv_biases_1)
        
	bn_conv1_mean, bn_conv1_variance = tf.nn.moments(h_conv1,[0,1,2,3])
        bn_conv1_scale = tf.Variable(tf.ones([32]))
        bn_conv1_offset = tf.Variable(tf.zeros([32]))
        bn_conv1_epsilon = 1e-3
	bn_conv1 = tf.nn.batch_normalization(h_conv1, bn_conv1_mean, bn_conv1_variance, bn_conv1_offset, bn_conv1_scale, bn_conv1_epsilon)
	

with tf.name_scope("conv2") as conv2:
        conv_weights_2 = weight_variable([2,2,32,64], "conv2_weights")
        conv_biases_2 = bias_variable([64], "conv2_biases")
        cw2_hist = tf.histogram_summary("conv2/weights", conv_weights_2)
        cb2_hist = tf.histogram_summary("conv2/biases", conv_biases_2)
        #c2 = tf.reshape(conv_weights_2, [32,64,2,2])
        #cw2_image_hist = tf.image_summary("conv2_w", c2)

        h_conv2 = tf.nn.relu(tf.nn.conv2d(bn_conv1, conv_weights_2, strides=[1, 2, 2, 1], padding="SAME") + conv_biases_2)

        bn_conv2_mean, bn_conv2_variance = tf.nn.moments(h_conv2, [0,1,2,3])
        bn_conv2_scale = tf.Variable(tf.ones([64]))
        bn_conv2_offset = tf.Variable(tf.zeros([64]))
        bn_conv2_epsilon = 1e-3
        bn_conv2 = tf.nn.batch_normalization(h_conv2, bn_conv2_mean, bn_conv2_variance, bn_conv2_offset, bn_conv2_scale, bn_conv2_epsilon)

        #h_pool2 = max_pool_2x2(h_conv2)


with tf.name_scope("fc_1") as fc_1:
	fc1_weights = weight_variable([5*1*64, 200], "fc1_weights")
        fc1_biases = bias_variable([200], "fc1_biases")
        fc1_b_hist = tf.histogram_summary("fc_1/biases", fc1_biases)
        fc1_w_hist = tf.histogram_summary("fc_1/weights", fc1_weights)

	h_pool3_flat = tf.reshape(bn_conv2, [-1,5*1*64])
	final_hidden_activation = tf.nn.relu(tf.matmul(h_pool3_flat, fc1_weights, name='final_hidden_activation') + fc1_biases)

with tf.name_scope("fc_2") as fc_2:
	fc2_weights = weight_variable([200, NUM_ACTIONS], "fc2_weights")
        fc2_biases = bias_variable([NUM_ACTIONS], "fc2_biases")
        fc2_w_hist = tf.histogram_summary("fc_2/weights", fc2_weights)
        fc2_b_hist = tf.histogram_summary("fc_2/biases", fc2_biases)

	output_layer = tf.matmul(final_hidden_activation, fc2_weights) + fc2_biases
	ol_hist = tf.histogram_summary("output_layer", output_layer)


with tf.name_scope("readout"):
	readout_action = tf.reduce_sum(tf.mul(output_layer, action), reduction_indices=1)
	r_hist = tf.histogram_summary("readout_action", readout_action)

with tf.name_scope("loss_summary"):
	loss = tf.reduce_mean(tf.square(target - readout_action))
        tf.scalar_summary("loss", loss)


merged = tf.merge_all_summaries()

sum_writer = tf.train.SummaryWriter('/tmp/train/c/', session.graph)

train_operation = tf.train.AdamOptimizer(0.001, epsilon=0.001).minimize(loss)

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

		state_from_env = get_current_state()
		reward = get_reward(state_from_env)
	
		#time.sleep(3)	
		##tkinter update
		global data
		data1 = np.asarray(state_from_env)
		data = np.reshape(data1, (2,10))
	
		#if we run for the first time, we build a state
		if first_time == 1:
			first_time = 0
			last_state = np.stack(tuple(state_from_env for _ in range(STATE_FRAMES)), axis=2)
			last_action = np.zeros([NUM_ACTIONS])  #speeed of both servos 0


		state_from_env = state_from_env.reshape(RESIZED_DATA_X, RESIZED_DATA_Y, 1)
		current_state = np.append(last_state[:,:,1:], state_from_env, axis=2)


		observations.append((last_state, last_action, reward, current_state))	

		if len(observations) > MEMORY_SIZE:
			observations.popleft()

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

		#if we got the max reward, we change degree_goal 
		if reward == 1:
			print("MAX REWARD -------- NEW DEGREE GOAL")
			global train_play_loop
			global probability_of_random_action
			global not_random
			global steps_done
			global steps_needed
			global accuracy
			global step

			print probability_of_random_action
			print train_play_loop
			
			print("steps_done1:",steps_done)
			print("steps_needed1",steps_needed)

			accuracy = accuracy + (steps_done - steps_needed)

#			if probability_of_random_action <= 0.0:
#				if step == 0:
#					accuracy = 0
#				if step >=  1000:
#					if accuracy > 0:
#						probability_of_random_action = 0.2
#				step += 1

			print "accuracy"	
			print accuracy
			if accuracy > 1000:
				accuracy = 0

			old = degree_goal
			print("old",old)
			degree_goal = random.randint(0, (RESIZED_DATA_X-1) )
			print("deg-goal:", degree_goal)
			if old > degree_goal:
				steps_needed = old - degree_goal
			elif degree_goal > old:
				steps_needed = degree_goal - old
			print("steps-needed:", steps_needed)
			if steps_needed == 0:
				steps_needed = 1
			steps_done = 0
			

			train_play_loop = 1
			if train_play_loop <= 0:

				t = raw_input("train or play? input 0 for play, number for how often it train and find degree_goal: ")
				t = int(t)
				if t == 0:
					nb = raw_input("new degree_goal: ")
					degree_goal = int(nb)
					print degree_goal
					#nb = raw_input("new probability of choosing random action: 0.0-1.0 : ")
					#probability_of_random_action = float(nb)
					#print probability_of_random_action
					train_play_loop = 1 #just to decrease it some steps later to 0 and not a negative number
				else:
					train_play_loop = t
					
			train_play_loop -= 1

		do_action(last_action)
	
except KeyboardInterrupt:
	print "save model"
	save_path = saver.save(session, "/home/ros/tensorflow-models/model-mini.ckpt")
	session.close()



