'''
Created on Jul 30, 2017

@author: ros
'''

import tensorflow as tf
import os.path
import numpy as np
import random
from math import sqrt
from keras.backend.tensorflow_backend import transpose
from tensorflow.python.ops import init_ops
import time
from tensorflow.python.training.training_util import global_step


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
        if kernel == 0:
            #kernel = self.conv_weights_1
            with tf.variable_scope("conv1", reuse=True):
                kernel = tf.get_variable("conv2d/kernel")
        if kernel == 1:
            kernel = self.conv_weights_2
        # get shape of the grid. NumKernels == grid_Y * grid_X
        def factorization(n):
            print("n>",n)
            print("fact>",int(sqrt(float(n))))
            for i in range(int(sqrt(float(n))), 0, -1):
                print("i>", i)
                if n % i == 0:
                    if i == 1: print('Who would enter a prime number of filters')
                    return (i, int(n / i))
        (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
        print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))
        
        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)
        #print("x_min>",tf.reduce_min(kernel).eval(session=self.session))
        #print("x_max>",tf.reduce_max(kernel).eval(session=self.session))
        kernel = (kernel - x_min) / (x_max - x_min)
        print("kernelshape>", kernel.get_shape())
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
        print("jojo>",self.agents_reward_per_action.shape)
        
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
        
        self.train_counter += 1
        
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
        #self.cw1_hist = tf.summary.histogram("conv1/weights", self.conv_weights_1)
        
        with tf.variable_scope("conv1", reuse=True):
            c = tf.get_variable("conv2d/bias")
            self.cb1_hist = tf.summary.histogram("conv1/biases", c)
            
            c1 = tf.get_variable("conv2d/kernel")
            self.cw1_hist = tf.summary.histogram("conv1/weights", c1)
        
        with tf.variable_scope("conv2", reuse=True):
            c2 = tf.get_variable("conv2d/bias")
            self.cw2_hist = tf.summary.histogram("conv2/biases", c2)
            c3 = tf.get_variable("conv2d/kernel")
            self.cb2_hist = tf.summary.histogram("conv2/weights", c3)
            
        with tf.variable_scope("fc_1", reuse=True):
            c4 = tf.get_variable("dense/bias")
            self.fc1_b_hist = tf.summary.histogram("fc_1/biases", c4)
            c5 = tf.get_variable("dense/kernel")
            self.fc1_w_hist = tf.summary.histogram("fc_1/weights", c5)
        
        self.fc2_w_hist = tf.summary.histogram("fc_2/weights", self.fc2_weights)
        self.fc2_b_hist = tf.summary.histogram("fc_2/biases", self.fc2_biases)
        
        self.r_hist = tf.summary.histogram("readout_action", self.readout_action)
        
        tf.summary.scalar("loss", self.loss)
        
        self.merged = tf.summary.merge_all()
    
        self.sum_writer = tf.summary.FileWriter('/tmp/train/'+self.timestamp, self.session.graph, flush_secs=30)
        
    def createFilterVisualization(self):
        grid = self.put_kernels_on_grid (0)
        tf.summary.image('conv1/filters1', grid, max_outputs=1)
        
        #grid1 = self.put_kernels_on_grid (1)
        #grid1 = tf.transpose(grid1, (1,2,3,0))
        #tf.summary.image('conv1/filters2', grid1, max_outputs=32)
        
        #grid2 = self.put_kernels_on_grid (1)
        #tf.summary.image('conv1/filters-fc1', grid2, max_outputs=1)
        
        #grid3 = self.put_kernels_on_grid (1)
        #tf.summary.image('conv1/filters-fc2', grid3, max_outputs=1)
    
    def createConvNet_1(self):
        with tf.name_scope("conv1") as conv1:
            with tf.variable_scope("conv1"):  
                self.h_conv1 = tf.layers.conv2d(self.input_layer, 12, [2,2],
                                                padding='same',
                                                activation=tf.nn.relu,
                                                bias_initializer=init_ops.TruncatedNormal(0.0, 0.01),
                                                kernel_initializer=init_ops.TruncatedNormal(0.0, 0.01))
                
                
            self.bn_conv1 = tf.layers.batch_normalization(self.h_conv1)
            
    def createConvNet_2(self):
        with tf.name_scope("conv2") as conv2:
            with tf.variable_scope("conv2"):
                self.h_conv2 = tf.layers.conv2d(self.bn_conv1, 24, [2,2],
                                                padding='same',
                                                activation=tf.nn.relu,
                                                bias_initializer=init_ops.TruncatedNormal(0.0, 0.01),
                                                kernel_initializer=init_ops.TruncatedNormal(0.0, 0.01),
                                                strides=(2,2))
            
            self.bn_conv2 = tf.layers.batch_normalization(self.h_conv2)
            #h_pool2 = max_pool_2x2(h_conv2)
    def createFCNet_1(self):
        with tf.name_scope("fc_1") as fc_1:
            with tf.variable_scope("fc_1"):
                self.h_pool3_flat = tf.reshape(self.bn_conv2, [-1,5*1*24])
                self.final_hidden_activation = tf.layers.dense(self.h_pool3_flat, 200,
                                                               activation=tf.nn.relu,
                                                               bias_initializer=init_ops.TruncatedNormal(0.0, 0.01),
                                                               kernel_initializer=init_ops.TruncatedNormal(0.0, 0.01))
        
    def createFCNet_2(self):
        with tf.name_scope("fc_2") as fc_2:
            with tf.variable_scope("fc_2"):
                self.fc2_weights = tf.Variable(tf.truncated_normal([200, self.NUM_ACTIONS], stddev=0.01))
                self.fc2_biases = tf.Variable(tf.constant(0.1, shape=[self.NUM_ACTIONS]))
            
                self.output_layer = tf.matmul(self.final_hidden_activation, self.fc2_weights) + self.fc2_biases
                self.ol_hist = tf.summary.histogram("output_layer", self.output_layer)
            
    def createReadOut(self):
        with tf.name_scope("readout"):
            self.readout_action = tf.reduce_sum(tf.multiply(self.output_layer, self.action), reduction_indices=1)
    
    def createLoss(self):
        with tf.name_scope("loss_summary"):
            #self.loss = tf.reduce_mean(tf.square(self.target - self.readout_action))
            self.loss = tf.reduce_mean(tf.abs(self.target - self.readout_action))
    
    def getSession(self):
        return self.session
    
    def saveSession(self):
        if self.train_counter % 100 == 3:
            self.save_path = self.saver.save(self.session, self.model_save_path+self.model_filename,
                                             global_step = self.train_counter)

    def getRewardAvg(self):
        return self.reward_avg
    
    def setRewardAvg(self,avg):
        self.reward_avg = avg
        print("setreward:",self.reward_avg)
    
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
        self.timestamp = time.strftime("%c")
        self.train_counter = 0
        self.model_save_path = "/home/ros/tensorflow-models/"+self.timestamp+"/"
        self.model_filename = "model.ckpt"
        self.restore_model_dir = ""  #change to dir name with stored model,e.g. m2/
        
        
        
        self.session =  tf.Session()
        self.action = tf.placeholder(tf.float32, [None, NUM_ACTIONS])
        self.target = tf.placeholder(tf.float32, [None])
        self.input_layer = tf.placeholder(tf.float32, [None, RESIZED_DATA_X, RESIZED_DATA_Y, STATE_FRAMES])
        
        self.sum_writer_index_var = tf.Variable(0, "sum_writer_index_var")
        self.add_sum_writer_index_var = self.sum_writer_index_var.assign(self.sum_writer_index_var + 1)
        
        self.reward_value = tf.Variable(0.0, "reward_value")
        self.reward_placeholder = tf.placeholder("float", [])
        self.assign_reward = self.reward_value.assign(self.reward_placeholder)
        self.reward_value_hist = tf.summary.scalar("reward_value", self.assign_reward)
        
        self.createConvNet_1()
        self.createConvNet_2()
        self.createFCNet_1()
        self.createFCNet_2()
        self.createReadOut()
        self.createLoss()   
        
            
        self.createFilterVisualization()    
        self.createHistogramSummaries()
        
              
        self.train_operation = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        if os.path.isfile(self.model_save_path+self.restore_model_dir+"checkpoint"):
            ckpt = tf.train.latest_checkpoint(self.model_save_path)
            self.saver.restore(self.session, ckpt)
            print "model restored"
        
            self.sum_writer_index = self.session.run(self.sum_writer_index_var)
