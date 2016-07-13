import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import tensorflow as tf
import Tkinter
from PIL import Image, ImageTk



max_degree = 264
degree_goal = 120
current_degree = 100
max_force = 1024
force_1_goal = 20
force_2_goal = 20
current_force_1 = 20
current_force_2 = 20


STATE_FRAMES = 4
NUM_ACTIONS = 3  #two speed values for two servos
NUM_STATES = 4624 # (2*max_degree) + (4*max_force)
MEMORY_SIZE = 40000
OBSERVATION_STEPS = 1000
MINI_BATCH_SIZE = 100
RESIZED_DATA_X = 68  #NUM_STATES resized to 68x68
RESIZED_DATA_Y = 68
FUTURE_REWARD_DISCOUNT = 0.9


probability_of_random_action = 1 
max_servo_speed_value = 400  #200 different speeds left, and 200 right
sum_writer_index = 0
train_play_loop = 0

data = None
photo = None
root = None
canvas = None


def get_current_state():
	global current_degree
	global current_force_1
	global current_force_2
	global degree_goal
	global force_1_goal
	global force_2_goal
	global max_degree
	global max_force


	a = np.zeros([max_degree])
	a[current_degree] = 1
	b = np.zeros([max_force])
	b[current_force_1] = 1
	c = np.zeros([max_force])
	c[current_force_2] = 1
	d = np.zeros([max_degree])
	d[degree_goal] = 1
	e = np.zeros([max_force])
	e[force_1_goal] = 1
	f = np.zeros([max_force])
	f[force_2_goal] = 1
	g = []
	g.extend(a)
	g.extend(b)
	g.extend(c)
	g.extend(d)
	g.extend(e)
	g.extend(f)
	h = np.reshape(g, (68,68))
	return h

#if we overlay, we get reward
def get_reward(current_state):
	global max_degree
	global max_force
	v = ( max_degree + (2*max_force) )
	s = np.reshape(current_state, (2, v))
	r = s[0] * s[1]
	r = sum(r)-2  #we dont change force, minus these two
	return r

#we choose a random or learned action
def choose_next_action(last_state):
	new_action = np.zeros([NUM_ACTIONS])
	global probability_of_random_action
	global max_servo_speed_value

	#simple decreaseing
	probability_of_random_action -= 0.000001
	
	if random.random() < probability_of_random_action:
		#new_action[0] = random.uniform(0, max_servo_speed_value)
		#new_action[1] = random.uniform(0, max_servo_speed_value)
		new_action_index = random.randint(0,2)
		new_action[new_action_index] = 1
		#print new_action
	else:
		readout_t = session.run(output_layer, feed_dict={input_layer: [last_state]})
		#print readout_t
		r1 = np.asarray(readout_t)
		r1 = np.reshape(r1, (NUM_ACTIONS))
#		if np.isnan(r1[0]) == True:
#			r1[0] = 0
#		if np.isnan(r1[1]) == True:
#			r1[1] = 0
#		
#		if r1[0] > 400:
#			r1[0] = 400
#			print("r1[0] action too high")
#		if r1[1] > 400:
#			r1[1] = 400
#			print("r1[1] action too high")
#
#		# ??? to prevent that the output_layer is producing NaN	
#		if (r1[0] < 0):
#			r1[0] = 0
#		if (r1[1] < 0):
#			r1[1] = 0
#			#print("a2", new_action)
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
def do_action(action):
	global force_1_goal
	global current_degree

	#if >= 200 means, servo direction forward, <200 means backward direction
#	if action[0] >= 200:
#		a = action[0] - 200
#		current_degree += a
#	elif action[0] < 200:
#		current_degree -= action[0]
#	elif action[1] >= 200:
#		a = action[1] - 200
#		current_degree += a
#	elif action[1] < 200:
#		current_degree -= action[1]
#
#	#end blocker
#	if current_degree > 263:
#		current_degree = 263
#	elif current_degree < 0:
#		current_degree = 0
	
	if action[0] == 1:
		current_degree = current_degree
	if action[1] == 1:
		current_degree += 1
	if action[2] == 1:
		current_degree -= 1

	if current_degree > 263:
        	current_degree = 263
        elif current_degree < 0:
        	current_degree = 0

	

def train(observations):
	print("train")
	global sum_writer_index

	mini_batch = random.sample(observations, MINI_BATCH_SIZE)
	previous_states = [d[0] for d in mini_batch]
        actions = [d[1] for d in mini_batch]
        rewards = [d[2] for d in mini_batch]
        current_states = [d[3] for d in mini_batch]

	agents_expected_reward = []

        agents_reward_per_action = session.run(output_layer, feed_dict={input_layer: current_states})

	for i in range(len(agents_reward_per_action)):
		if np.isnan(agents_reward_per_action[i][0]) == True:
			print "NNNNNNNNNNNNNNAAAAAAAAAAAANNNNNNNNNNN"
			agents_reward_per_action[i][0] = 0
		if np.isnan(agents_reward_per_action[i][1]) == True:
			print "NNNNNNNNNNNNAAAAANNNNNNNNNNNNNNNN"
			agents_reward_per_action[i][1] = 0

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
        canvas.create_image(0,0,image=photo,anchor=Tkinter.NW)
        root.update()

        root.after(100,image_loop)



######  main python program starts here  #####

observations = deque()
first_time = 1
last_state = None


#######TK inter 
root = Tkinter.Tk()
frame = Tkinter.Frame(root, width=68, height=68)
frame.pack()
canvas = Tkinter.Canvas(frame, width=68,height=68)
#canvas.place(x=-2,y=-2)
canvas.place(x=0,y=0)
root.after(1000,image_loop) # INCREASE THE 0 TO SLOW IT DOWN


#################### create network

session = tf.Session()

state = tf.placeholder("float", [None, NUM_STATES])
action = tf.placeholder("float", [None, NUM_ACTIONS])
target = tf.placeholder("float", [None])

with tf.name_scope("conv1") as conv1:
	conv_weights_1 = weight_variable([8,8,4,32], "conv1_weights")
        conv_biases_1 = bias_variable([32], "conv1_biases")
        cw1_hist = tf.histogram_summary("conv1/weights", conv_weights_1)
        cb1_hist = tf.histogram_summary("conv1/biases", conv_biases_1)
        c1 = tf.reshape(conv_weights_1, [32, 8,8, 4])
        cw1_image_hist = tf.image_summary("conv1_w", c1)

with tf.name_scope("conv2") as conv2:
        conv_weights_2 = weight_variable([4,4,32,64], "conv2_weights")
        conv_biases_2 = bias_variable([64], "conv2_biases")
        cw2_hist = tf.histogram_summary("conv2/weights", conv_weights_2)
        cb2_hist = tf.histogram_summary("conv2/biases", conv_biases_2)
        c2 = tf.reshape(conv_weights_2, [32,64,4,4])
        cw2_image_hist = tf.image_summary("conv2_w", c2)

with tf.name_scope("conv3") as conv3:
        conv_weights_3 = weight_variable([3,3,64,64], "conv3_weights")
        conv_biases_3 = bias_variable([64], "conv3_biases")
        cw3_hist = tf.histogram_summary("conv3/weights", conv_weights_3)
        cb3_hist = tf.histogram_summary("conv3/biases", conv_biases_3)
        c3 = tf.reshape(conv_weights_3, [64,64,3,3])
        cw3_image_hist = tf.image_summary("conv3_w", c3)

with tf.name_scope("fc_1") as fc_1:
        fc1_weights = weight_variable([2*2*64, 4624], "fc1_weights")
        fc1_biases = bias_variable([4624], "fc1_biases")
        fc1_b_hist = tf.histogram_summary("fc_1/biases", fc1_biases)
        fc1_w_hist = tf.histogram_summary("fc_1/weights", fc1_weights)

with tf.name_scope("fc_2") as fc_2:
        fc2_weights = weight_variable([4624, NUM_ACTIONS], "fc2_weights")
        fc2_biases = bias_variable([NUM_ACTIONS], "fc2_biases")
        fc2_w_hist = tf.histogram_summary("fc_2/weights", fc2_weights)
        fc2_b_hist = tf.histogram_summary("fc_2/biases", fc2_biases)

input_layer = tf.placeholder("float", [None, RESIZED_DATA_X, RESIZED_DATA_Y, STATE_FRAMES])

h_conv1 = tf.nn.relu(tf.nn.conv2d(input_layer, conv_weights_1, strides=[1, 4, 4, 1], padding="SAME") + conv_biases_1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, conv_weights_2, strides=[1, 2, 2, 1], padding="SAME") + conv_biases_2)
h_pool2 = max_pool_2x2(h_conv2)


h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, conv_weights_3, strides=[1,1,1,1], padding="SAME") + conv_biases_3)
h_pool3 = max_pool_2x2(h_conv3)

h_pool3_flat = tf.reshape(h_pool3, [-1,2*2*64])
final_hidden_activation = tf.nn.relu(tf.matmul(h_pool3_flat, fc1_weights, name='final_hidden_activation') + fc1_biases)

output_layer = tf.matmul(final_hidden_activation, fc2_weights) + fc2_biases
ol_hist = tf.histogram_summary("output_layer", output_layer)


#we feed in the action the NN would do and targets=rewards ???
readout_action = tf.reduce_sum(tf.mul(output_layer, action), reduction_indices=1)
r_hist = tf.histogram_summary("readout_action", readout_action)

with tf.name_scope("loss_summary"):
	#loss = tf.reduce_mean(tf.square(output - target))
	loss = tf.reduce_mean(tf.square(target - readout_action))
        #loss = tf.reduce_mean(tf.square(output_layer - target))
        tf.scalar_summary("loss", loss)

merged = tf.merge_all_summaries()

sum_writer = tf.train.SummaryWriter('/tmp/train', session.graph)

train_operation = tf.train.AdamOptimizer(0.1).minimize(loss)

session.run(tf.initialize_all_variables())
saver = tf.train.Saver()




########### end create network


data=np.array(np.random.random((68,68))*100,dtype=int)


try:
	while True:
		#tkinter update
		root.update_idletasks()
		root.update()

		#print("test if it loops or blocks")
		state_from_env = get_current_state()
		reward = get_reward(state_from_env)

		##tkinter update
		global data
		#data=np.array(np.random.random((68,68))*100,dtype=int)
		data=state_from_env
		#data = np.reshape(state_from_env, (68,68,1))
		data = data * 255

	
		#if we run for the first time, we build a state
		if first_time == 1:
			first_time = 0
			last_state = np.stack(tuple(state_from_env for _ in range(STATE_FRAMES)), axis=2)
			last_action = np.zeros([NUM_ACTIONS])  #speeed of both servos 0
			#do_action() which does nothing


		state_from_env = state_from_env.reshape(68,68,1)
		current_state = np.append(last_state[:,:,1:], state_from_env, axis=2)

		observations.append((last_state, last_action, reward, current_state))	

		if len(observations) > MEMORY_SIZE:
			#print("POPLEFT---------------------")
			observations.popleft()

		#print len(observations)
		if len(observations) % OBSERVATION_STEPS == 0:
			train(observations)

		last_state = current_state
		last_action = choose_next_action(last_state)

		#if we got the max reward, we change degree_goal mostly, perhaps sometimes force_1/2_goal
		if reward == 1:
			print("MAX REWARD -------- NEW DEGREE GOAL")
			global train_play_loop
			global probability_of_random_action
			
			print probability_of_random_action
			print train_play_loop

			if train_play_loop == 0:
				t = raw_input("train or play? input 0 for play, number for how often it train and find degree_goal")
				t = int(t)
				if t == 0:
					nb = raw_input("new degree_goal:")
					degree_goal = int(nb)
					print degree_goal
					nb = raw_input("new probability of choosing random action:0.0-1.0")
					probability_of_random_action = float(nb)
					print probability_of_random_action
					train_play_loop = 1 #just to decrease it some steps later to 0 and not a negative number
				else:
					degree_goal = random.randint(0, (max_degree-1) )
					train_play_loop = t		
			train_play_loop -= 1

		do_action(last_action)
	
except KeyboardInterrupt:
	print "save model"
	save_path = saver.save(session, "/home/ros/tensorflow-models/model-mini.ckpt")
	session.close()



