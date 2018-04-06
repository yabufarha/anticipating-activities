#!/usr/bin/python2.7

import tensorflow as tf
import numpy as np
import os
from scipy.ndimage.filters import gaussian_filter1d
from utils.helper_functions import get_label_length_seq

class ModelCNN:
    
    def __init__(self, nRows, nCols):
        self.input_vid = tf.placeholder('float', [None, nRows, nCols, 1], name='input_vid')
        self.target = tf.placeholder('float', [None, nRows, nCols, 1], name='target')
        self.nRows = nRows
        self.nCols = nCols
        
        self.__build()
   

    def __weight_variable(self, shape, myName):
        initial = tf.truncated_normal(shape, stddev=0.1, name=myName)
        return tf.Variable(initial)
    
    
    def __bias_variable(self, shape, myName):
        initial = tf.constant(0.1, shape=shape, name=myName)
        return tf.Variable(initial)
    
    
    def __conv(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
        
        
    def __max_pool_2x1(self, x):
      return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')
      
      
    def __build(self):
        w_conv1 = self.__weight_variable([5, 1, 1, 8], 'w_conv1') 
        b_conv1 = self.__bias_variable([8], 'b_conv1')
        w_conv2 = self.__weight_variable([5, 1, 8, 16], 'w_conv2') 
        b_conv2 = self.__bias_variable([16], 'b_conv2')    
        W_fc1 = self.__weight_variable([int(4*1*self.nRows*self.nCols), 1024], 'W_fc1')
        b_fc1 = self.__bias_variable([1024], 'b_fc1')
        W_fc2 = self.__weight_variable([1024, self.nRows*self.nCols], 'W_fc2')
        b_fc2 = self.__bias_variable([self.nRows*self.nCols], 'b_fc2')
        
        h_conv1 = tf.nn.relu(self.__conv(self.input_vid, w_conv1) + b_conv1)
        h_pool1 = self.__max_pool_2x1(h_conv1)
    
        h_conv2 = tf.nn.relu(self.__conv(h_pool1, w_conv2) + b_conv2)
        h_pool2 = self.__max_pool_2x1(h_conv2)
    
        input_vid_flat = tf.reshape(h_pool2, [-1, int(4*1*self.nRows*self.nCols)])
        h_fc1 = tf.nn.relu(tf.matmul(input_vid_flat, W_fc1) + b_fc1)

        pred_flat = tf.matmul(h_fc1, W_fc2) + b_fc2
        prediction_unscaled = tf.reshape(pred_flat, [-1, self.nRows, self.nCols, 1])
        ## l2 normalization
        self.prediction = tf.nn.l2_normalize(prediction_unscaled, dim=2)
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=100)


    def train(self, sess, model_save_path, batch_gen, nEpochs, save_freq, batch_size):
        my_loss = tf.reduce_mean(tf.square( self.target - self.prediction ))
        correct_prediction = tf.equal(tf.argmax(self.prediction,2), tf.argmax(self.target,2))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
        optimizer = tf.train.AdamOptimizer(0.001).minimize(my_loss) 
                
        sess.run(tf.global_variables_initializer())
                       
        for epoch in range(nEpochs):
            epoch_acc = 0
            i=0
            while(batch_gen.has_next()):
                batch_vid, batch_target = batch_gen.next_batch(batch_size)
                _, acc = sess.run([optimizer, accuracy], feed_dict={self.input_vid: batch_vid, self.target: batch_target})
                i=i+1
                epoch_acc += acc
            batch_gen.reset()
            
            if epoch%save_freq==0:  
                print 'Epoch', (epoch+1), 'completed out of',nEpochs,'training Acc: %.2f'%(epoch_acc/i)
                path = model_save_path+"/epoch-"+str(epoch+1)
                if not os.path.exists(path):
                    os.makedirs(path)
                self.saver.save(sess, path+"/model.ckpt")
    

    def __post_process(self, result, sigma):
        new_res =  gaussian_filter1d(result, sigma=sigma, axis=0)
        return new_res

    
    def predict(self, sess, model_save_path, input_x, sigma, actions_dict):
            self.saver.restore(sess, model_save_path)
            result = sess.run([self.prediction], feed_dict={self.input_vid: input_x})[0]      
            result = np.reshape(result,[self.nRows, self.nCols])
            result = self.__post_process(result, sigma)
            output = []
            for i in range(len(result)):
                output.append(actions_dict.keys()[actions_dict.values().index(np.argmax(result[i]))])
            label_seq, length_seq = get_label_length_seq(output)
            return label_seq, length_seq
