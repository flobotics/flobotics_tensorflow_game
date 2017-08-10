'''
Created on Jul 30, 2017

@author: ros
'''

import tensorflow as tf
import os.path
import numpy as np
import random
from math import sqrt


class tFParams():
    def put_kernels_on_grid (self, kernel, pad = 1):

        '''Visualize conv. filters as an image (mostly for the 1st layer).
        Arranges filters into a grid, with some paddings between adjacent filters.
        Args:
          kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
          pad:               number of black pixels around each filter (between them)
        Return:
          Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
        '''
        
        kernel = self.conv_weights_1
        # get shape of the grid. NumKernels == grid_Y * grid_X
        def factorization(n):
          for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
              if i == 1: print('Who would enter a prime number of filters')
              return (i, int(n / i))
        (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
        print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))
        
        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)
        kernel = (kernel - x_min) / (x_max - x_min)
        
        # pad X and Y
        x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')
        
        # X and Y dimensions, w.r.t. padding
        Y = kernel.get_shape()[0] + 2 * pad
        X = kernel.get_shape()[1] + 2 * pad
        
        channels = kernel.get_shape()[2]
        
        # put NumKernels to the 1st dimension
        x = tf.transpose(x, (3, 0, 1, 2))
        # organize grid on Y axis
        x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))
        
        # switch X and Y axes
        x = tf.transpose(x, (0, 2, 1, 3))
        # organize grid on X axis
        x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))
        
        # back to normal order (not combining with the next step for clarity)
        x = tf.transpose(x, (2, 1, 3, 0))
        
        # to tf.image_summary order [batch_size, height, width, channels],
        #   where in this case batch_size == 1
        x = tf.transpose(x, (3, 0, 1, 2))
        
        # scaling to [0, 255] is not necessary for tensorboard
        return x 

    
    
    def train(self, observations):
        print("train")
    
        self.mini_batch = random.sample(observations, self.MINI_BATCH_SIZE)
        self.previous_states = [d[0] for d in self.mini_batch]
        self.actions = [d[1] for d in self.mini_batch]
        self.rewards = [d[2] for d in self.mini_batch]
        self.current_states = [d[3] for d in self.mini_batch]
    
        self.agents_expected_reward = []
        self.agents_reward_per_action = self.session.run(self.output_layer, feed_dict={self.input_layer: self.current_states})
        
        if self.reward_avg_count > 0:
            self.reward_avg = self.reward_avg / self.reward_avg_count
        rew = np.array(self.reward_avg)
        self.reward_avg = 0
        self.reward_avg_count = 0
    
        for i in range(len(self.mini_batch)):
            self.agents_expected_reward.append(self.rewards[i] + self.FUTURE_REWARD_DISCOUNT * np.max(self.agents_reward_per_action[i]))
    
        _, __, self.result = self.session.run([self.train_operation, self.assign_reward, self.merged], feed_dict={self.reward_placeholder: rew, self.input_layer: self.previous_states, self.action : self.actions, self.target: self.agents_expected_reward})
        
        self.sum_writer.add_summary(self.result, self.sum_writer_index)
        self.sum_writer_index += 1
    
        self.session.run(self.add_sum_writer_index_var)
        
    #we choose a random or learned action
    def choose_next_action(self, last_state, probability_of_random_action):   
        new_action = np.zeros([self.NUM_ACTIONS])
        
        if random.random() < probability_of_random_action:
            new_action_index = random.randint(0,2)
            new_action[new_action_index] = 1
            #print new_action
        else:
            readout_t = self.session.run(self.output_layer, feed_dict={self.input_layer: [last_state]})
            r1 = np.asarray(readout_t)
            r1 = np.reshape(r1, (self.NUM_ACTIONS))
            action_index = np.argmax(readout_t)
            new_action[action_index] = 1
            #print new_action
        
        return new_action
    
    def createHistogramSummaries(self):
        self.cw1_hist = tf.histogram_summary("conv1/weights", self.conv_weights_1)
        self.cb1_hist = tf.histogram_summary("conv1/biases", self.conv_biases_1)
        
        self.cw2_hist = tf.histogram_summary("conv2/weights", self.conv_weights_2)
        self.cb2_hist = tf.histogram_summary("conv2/biases", self.conv_biases_2)
        
        self.fc1_b_hist = tf.histogram_summary("fc_1/biases", self.fc1_biases)
        self.fc1_w_hist = tf.histogram_summary("fc_1/weights", self.fc1_weights)
        
        self.fc2_w_hist = tf.histogram_summary("fc_2/weights", self.fc2_weights)
        self.fc2_b_hist = tf.histogram_summary("fc_2/biases", self.fc2_biases)
        
        self.r_hist = tf.histogram_summary("readout_action", self.readout_action)
        
        tf.scalar_summary("loss", self.loss)
        
        self.merged = tf.merge_all_summaries()
        
        self.sum_writer = tf.train.SummaryWriter('/tmp/train/c/', self.session.graph)
    
    def getSession(self):
        return self.session
    
    def saveSession(self):
        self.save_path = self.saver.save(self.session, "/home/ros/tensorflow-models/model-mini.ckpt")

    def getRewardAvg(self):
        return self.reward_avg
    
    def setRewardAvg(self,avg):
        self.reward_avg = avg
    
    def getRewardAvgCount(self):
        return self.reward_avg_count
    
    def setRewardAvgCount(self,avgC):
        self.reward_avg_count = avgC
        
    def __init__(self, NUM_ACTIONS=3, RESIZED_DATA_X=10, RESIZED_DATA_Y=2, STATE_FRAMES=4, MINI_BATCH_SIZE=100, OBSERVATION_STEPS=500):
        '''
        Constructor
        '''
        self.OBSERVATION_STEPS = OBSERVATION_STEPS
        self.FUTURE_REWARD_DISCOUNT = 0.90
        self.NUM_ACTIONS = NUM_ACTIONS
        self.MINI_BATCH_SIZE = MINI_BATCH_SIZE
        self.reward_avg_count = 0
        self.reward_avg = 0
        self.sum_writer_index = 0
        
        
        self.session =  tf.Session()
        self.action = tf.placeholder("float", [None, NUM_ACTIONS])
        self.target = tf.placeholder("float", [None])
        self.input_layer = tf.placeholder("float", [None, RESIZED_DATA_X, RESIZED_DATA_Y, STATE_FRAMES])
        
        self.sum_writer_index_var = tf.Variable(0, "sum_writer_index_var")
        self.add_sum_writer_index_var = self.sum_writer_index_var.assign(self.sum_writer_index_var + 1)
        
        self.reward_value = tf.Variable(0.0, "reward_value")
        self.reward_placeholder = tf.placeholder("float", [])
        self.assign_reward = self.reward_value.assign(self.reward_placeholder)
        self.reward_value_hist = tf.scalar_summary("reward_value", self.assign_reward)
        
        with tf.name_scope("conv1") as conv1:
            self.conv_weights_1 = tf.Variable(tf.truncated_normal([10, 2, 4,32], stddev=0.01))
            self.conv_biases_1 = tf.Variable(tf.constant(0.1, shape=[32]))
            
            
            
            a = tf.nn.conv2d(self.input_layer, self.conv_weights_1, strides=[1, 1, 1, 1], padding="SAME")
            grid = self.put_kernels_on_grid (a)
            tf.summary.image('conv1/kernels', grid, max_outputs=1)
            self.h_conv1 = tf.nn.relu(a + self.conv_biases_1)
                
            self.bn_conv1_mean, self.bn_conv1_variance = tf.nn.moments(self.h_conv1,[0,1,2,3])
            self.bn_conv1_scale = tf.Variable(tf.ones([32]))
            self.bn_conv1_offset = tf.Variable(tf.zeros([32]))
            self.bn_conv1_epsilon = 1e-3
            self.bn_conv1 = tf.nn.batch_normalization(self.h_conv1, self.bn_conv1_mean, self.bn_conv1_variance, self.bn_conv1_offset, self.bn_conv1_scale, self.bn_conv1_epsilon)
            
        
        with tf.name_scope("conv2") as conv2:
            self.conv_weights_2 =tf.Variable(tf.truncated_normal([2,2,32,64], stddev=0.01))
            self.conv_biases_2 = tf.Variable(tf.constant(0.1, shape=[64]))

            self.h_conv2 = tf.nn.relu(tf.nn.conv2d(self.bn_conv1, self.conv_weights_2, strides=[1, 2, 2, 1], padding="SAME") + self.conv_biases_2)
    
            self.bn_conv2_mean, self.bn_conv2_variance = tf.nn.moments(self.h_conv2, [0,1,2,3])
            self.bn_conv2_scale = tf.Variable(tf.ones([64]))
            self.bn_conv2_offset = tf.Variable(tf.zeros([64]))
            self.bn_conv2_epsilon = 1e-3
            self.bn_conv2 = tf.nn.batch_normalization(self.h_conv2, self.bn_conv2_mean, self.bn_conv2_variance, self.bn_conv2_offset, self.bn_conv2_scale, self.bn_conv2_epsilon)
    
            #h_pool2 = max_pool_2x2(h_conv2)
        
        
        with tf.name_scope("fc_1") as fc_1:
            self.fc1_weights = tf.Variable(tf.truncated_normal([5*1*64, 200], stddev=0.01))
            self.fc1_biases = tf.Variable(tf.constant(0.1, shape=[200]))
        
            self.h_pool3_flat = tf.reshape(self.bn_conv2, [-1,5*1*64])
            self.final_hidden_activation = tf.nn.relu(tf.matmul(self.h_pool3_flat, self.fc1_weights, name='final_hidden_activation') + self.fc1_biases)
        
        with tf.name_scope("fc_2") as fc_2:
            self.fc2_weights = tf.Variable(tf.truncated_normal([200, NUM_ACTIONS], stddev=0.01))
            self.fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_ACTIONS]))
        
            self.output_layer = tf.matmul(self.final_hidden_activation, self.fc2_weights) + self.fc2_biases
            self.ol_hist = tf.histogram_summary("output_layer", self.output_layer)
        
        
        with tf.name_scope("readout"):
            self.readout_action = tf.reduce_sum(tf.mul(self.output_layer, self.action), reduction_indices=1)
            
        
        with tf.name_scope("loss_summary"):
            self.loss = tf.reduce_mean(tf.square(self.target - self.readout_action))
            
            
        self.createHistogramSummaries()
              
        self.train_operation = tf.train.AdamOptimizer(0.0001, epsilon=0.001).minimize(self.loss)
        
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
        if os.path.isfile("/home/ros/tensorflow-models/model-mini.ckpt"):
            self.saver.restore(self.session, "/home/ros/tensorflow-models/model-mini.ckpt")
            print "model restored"
        
            self.sum_writer_index = self.session.run(self.sum_writer_index_var)
