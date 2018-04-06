#!/usr/bin/python2.7

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
    
class ModelRNN:
    
    def __init__(self, nClasses, rnn_size, max_seq_sz, num_layers):
        self.input_seq = tf.placeholder('float', [None, max_seq_sz, nClasses+1])
        self.target = tf.placeholder('float', [None, nClasses+2])
        self.nClasses = nClasses
        self.rnn_size = rnn_size
        self.max_seq_sz = max_seq_sz
        self.num_layers = num_layers
        
        self.__build()
    
    
    def __weight_variable(self, shape, myName):
        initial = tf.random_normal(shape, stddev=0.1, name=myName)
        return tf.Variable(initial)
    
    
    def __bias_variable(self, shape, myName):
        initial = tf.constant(0.1, shape=shape, name=myName)
        return tf.Variable(initial)
        
        
    def __build(self):
        w_fc_in = self.__weight_variable([self.nClasses+1, 128], 'w_fc_in')
        b_fc_in = self.__bias_variable([128], 'b_fc_in')
        
        w_fc_o = self.__weight_variable([self.rnn_size, 128], 'w_fc_o')
        b_fc_o = self.__bias_variable([128], 'b_fc_o')
                
        w_output_action = self.__weight_variable([128, self.nClasses], 'w_fc_in')
        b_output_action = self.__bias_variable([self.nClasses], 'b_fc_in')
        
        w_output_len = self.__weight_variable([128, 2], 'w_fc_in')
        b_output_len = self.__bias_variable([2], 'b_fc_in')
        
        x = tf.reshape(self.input_seq, [-1, self.nClasses+1])
        h1 = tf.nn.relu(tf.matmul(x, w_fc_in) + b_fc_in)
        h1 = tf.reshape(h1, [-1,self.max_seq_sz,128])
        #rnn
        h1 = tf.unstack(h1, axis=1)
        def get_cell():
            return rnn.GRUCell(self.rnn_size)   
        gru_cell = rnn.MultiRNNCell([get_cell() for _ in range(self.num_layers)])
        outputs, states = rnn.static_rnn(gru_cell, h1, dtype=tf.float32) 
        #fc_o
        h2 = tf.nn.relu(tf.matmul(outputs[-1], w_fc_o) + b_fc_o)
        #output
        output_label = tf.matmul(h2, w_output_action) + b_output_action
        output_len = tf.nn.relu(tf.matmul(h2, w_output_len) + b_output_len)
        #    
        self.prediction = tf.concat([output_label, output_len], 1)
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=100)
        
        
    def train(self, sess, model_save_path, batch_gen, nEpochs, save_freq, batch_size):
        gt_labels = self.target[:,:-2]
        gt_length = self.target[:,-2:]
        predicted_labels = self.prediction[:,:-2]
        predicted_length = self.prediction[:,-2:]
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt_labels, logits=predicted_labels, dim=1))
        loss += tf.reduce_mean(tf.square( gt_length - predicted_length ))
        
        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
                
        sess.run(tf.global_variables_initializer())
      
        for epoch in range(nEpochs):
            epoch_loss = 0
            i=0
            while(batch_gen.has_next()):
                batch_in, batch_target = batch_gen.next_batch(batch_size)
                _, err = sess.run([optimizer, loss], feed_dict={self.input_seq: batch_in, self.target: batch_target})
                i=i+1
                epoch_loss += err       
            batch_gen.reset()
            
            if epoch%save_freq==0:  
                print 'Epoch', (epoch+1), 'completed out of',nEpochs,'epoch loss: %.2f'%(epoch_loss/i)
                path = model_save_path+"/epoch-"+str(epoch+1)
                if not os.path.exists(path):
                    os.makedirs(path)
                self.saver.save(sess, path+"/model.ckpt")
                    
    
    def predict(self, sess, model_save_path, pred_len, label_seq, length_seq, actions_dict, T):
        self.saver.restore(sess, model_save_path)
        l = 0
        while l < pred_len:
            p_seq = np.zeros((self.max_seq_sz, self.nClasses+1))
            for i in range(len(label_seq[-self.max_seq_sz:])):
                p_seq[i][-1] = length_seq[i]/T
                p_seq[i][ actions_dict[label_seq[i]] ] = 1
    
            result = self.prediction.eval({self.input_seq:[p_seq]})[0]
            
            if int(result[-1]*T) > 0:
                length_seq[-1] += result[-1]*T
                l = l + int(result[-1]*T)
            if int(result[-2]*T) > 0:
                l = l + int(result[-2]*T)
                label_seq.append(actions_dict.keys()[actions_dict.values().index(np.argmax(result[:-2]))])
                length_seq.append(result[-2]*T)
            if int(result[-1]*T) == 0 and int(result[-2]*T) == 0:
                l = l+pred_len
                length_seq[-1] += pred_len
    
        return label_seq, length_seq
    