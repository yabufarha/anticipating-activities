#!/usr/bin/python2.7

class Base_batch_generator(object):
    
    def __init__(self):
        self.list_of_examples = []
        self.index = 0
    
    def number_of_examples(self):
        return len(self.list_of_examples)
        
    def reset(self):
        self.index = 0
        return
        
    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False
    
    def read_data(self, list_of_videos):
        raise NotImplementedError()
    
    def next_batch(self, batch_size):
        raise NotImplementedError()