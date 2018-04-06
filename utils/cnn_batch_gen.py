#!/usr/bin/python2.7

import numpy as np
import random
from helper_functions import encode_content
from base_batch_gen import Base_batch_generator

class CNN_batch_generator(Base_batch_generator):
    
    def __init__(self, nRows, nCols, actions_dict):
        super(CNN_batch_generator, self).__init__()
        self.nRows = nRows
        self.nCols = nCols
        self.actions_dict = actions_dict
        
    def read_data(self, list_of_videos):
        for vid in list_of_videos:
            file_ptr = open(vid, 'r') 
            content = file_ptr.read().split('\n')[:-1] 
            
            obs_perc = [.1, .2, .3, .5]
    
            for i in range(len(obs_perc)):
                observed_content = content[:int(obs_perc[i]*len(content))]
                input_vid = encode_content(observed_content, self.nRows, self.nCols, self.actions_dict)
                input_vid = np.reshape(input_vid, [self.nRows, self.nCols, 1])
                
                target_content = content[int(obs_perc[i]*len(content)):int((0.5+obs_perc[i])*len(content))]
                target = encode_content(target_content, self.nRows, self.nCols, self.actions_dict)
                target = np.reshape(target, [self.nRows, self.nCols, 1])
                example = [input_vid, target]
                self.list_of_examples.append(example)
        random.shuffle(self.list_of_examples) 
        return
            
    def next_batch(self, batch_size):        
        batch = np.array(self.list_of_examples[self.index:self.index+batch_size])
        self.index += batch_size
        batch_vid = list(batch[:,0])
        batch_target = list(batch[:,1])
        
        return batch_vid, batch_target

