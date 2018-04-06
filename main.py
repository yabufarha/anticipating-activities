#!/usr/bin/python2.7

import tensorflow as tf
import numpy as np
import argparse
from models.cnn import ModelCNN
from models.rnn import ModelRNN
from utils.base_batch_gen import Base_batch_generator
from utils.rnn_batch_gen import RNN_batch_generator
from utils.cnn_batch_gen import CNN_batch_generator
from utils.helper_functions import read_mapping_dict, encode_content, write_predictions, get_label_length_seq

parser = argparse.ArgumentParser()

parser.add_argument("--model", default="rnn", help="select model: [\"rnn\", \"cnn\"]")
parser.add_argument("--action", default="predict", help="select action: [\"train\", \"predict\"]")

parser.add_argument("--mapping_file", default="./data/mapping_bf.txt")
parser.add_argument("--vid_list_file", default="./data/test.split1.bundle")
parser.add_argument("--model_save_path", default="./save_dir/models/rnn")
parser.add_argument("--results_save_path", default="./save_dir/results/rnn")

parser.add_argument("--save_freq", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--nEpochs", type=int, default=20)
parser.add_argument("--eval_epoch", type=int, default=20)

#RNN specific parameters
parser.add_argument("--rnn_size", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--max_seq_sz", type=int, default=25)
parser.add_argument("--alpha", type=float, default=6, help="a scalar value used in normalizing the input length")
parser.add_argument("--n_iterations", type=int, default=10, help="number of training examples corresponding to each action segment for the rnn")

#CNN specific parameters
parser.add_argument("--nRows", type=int, default=128)
parser.add_argument("--sigma", type=int, default=3, help="sigma for the gaussian smoothing step")

#Test on GT or decoded input
parser.add_argument("--input_type", default="decoded", help="select input type: [\"decoded\", \"gt\"]")
parser.add_argument("--decoded_path", default="./data/decoded/split1")

################################################################################################################################################

args, unknown = parser.parse_known_args()

actions_dict = read_mapping_dict(args.mapping_file)
nClasses = len(actions_dict)

file_ptr = open(args.vid_list_file, 'r') 
list_of_videos = file_ptr.read().split('\n')[1:-1]

################
# Training #####
################
if args.action == "train":
    model = None
    batch_gen = Base_batch_generator()
    
    if args.model == "rnn":
        model = ModelRNN(nClasses, args.rnn_size, args.max_seq_sz, args.num_layers)
        batch_gen = RNN_batch_generator(nClasses, args.n_iterations, args.max_seq_sz, actions_dict, args.alpha)
    elif args.model == "cnn":
        model = ModelCNN(args.nRows, nClasses)
        batch_gen = CNN_batch_generator(args.nRows, nClasses, actions_dict)
        
    batch_gen.read_data(list_of_videos)
    with tf.Session() as sess:
        model.train(sess, args.model_save_path, batch_gen, args.nEpochs, args.save_freq, args.batch_size)

##################
# Prediction #####
##################
elif args.action == "predict":
    pred_percentages = [.1, .2, .3, .5]
    obs_percentages = [.2, .3]
    model_restore_path = args.model_save_path+"/epoch-"+str(args.eval_epoch)+"/model.ckpt" 
    
    if args.model == "rnn":
        model = ModelRNN(nClasses, args.rnn_size, args.max_seq_sz, args.num_layers)
        for vid in list_of_videos:
            f_name = vid.split('/')[-1].split('.')[0]
            observed_content=[]
            vid_len = 0
            if args.input_type == "gt":
                file_ptr = open(vid, 'r') 
                content = file_ptr.read().split('\n')[:-1] 
                vid_len = len(content)
                
            for obs_p in obs_percentages:
                
                if args.input_type == "decoded":
                    file_ptr = open(args.decoded_path+"/obs"+str(obs_p)+"/"+f_name+'.txt', 'r') 
                    observed_content = file_ptr.read().split('\n')[:-1]
                    vid_len = int(len(observed_content)/obs_p)
                elif args.input_type == "gt":
                    observed_content = content[:int(obs_p*vid_len)]
                T = (1.0/args.alpha)*vid_len
                
                for pred_p in pred_percentages:
                    pred_len = int(pred_p*vid_len)  
                    output_len = pred_len + len(observed_content)
                    
                    label_seq, length_seq = get_label_length_seq(observed_content)                    
                    with tf.Session() as sess:
                        label_seq, length_seq = model.predict(sess, model_restore_path, pred_len, label_seq, length_seq, actions_dict, T)
                    
                    recognition = []
                    for i in range(len(label_seq)):
                        recognition = np.concatenate((recognition, [label_seq[i]]*int(length_seq[i])))
                    recognition = recognition[:output_len]
                    #write results to file
                    f_name = vid.split('/')[-1].split('.')[0]
                    path=args.results_save_path+"/obs"+str(obs_p)+"-pred"+str(pred_p)
                    write_predictions(path, f_name, recognition)
                    
                        
    elif args.model == "cnn":
        model = ModelCNN(args.nRows, nClasses)
        for vid in list_of_videos:
            f_name = vid.split('/')[-1].split('.')[0]
            observed_content=[]
            vid_len = 0
            if args.input_type == "gt":
                file_ptr = open(vid, 'r') 
                content = file_ptr.read().split('\n')[:-1] 
                vid_len = len(content)
                
            for obs_p in obs_percentages:
                if args.input_type == "decoded":
                    file_ptr = open(args.decoded_path+"/obs"+str(obs_p)+"/"+f_name+'.txt', 'r') 
                    observed_content = file_ptr.read().split('\n')[:-1]
                    vid_len = int(len(observed_content)/obs_p)
                elif args.input_type == "gt":
                    observed_content = content[:int(obs_p*vid_len)]
                
                input_x = encode_content(observed_content, args.nRows, nClasses, actions_dict)
                input_x = [np.reshape(input_x, [args.nRows, nClasses, 1])]
                
                with tf.Session() as sess:
                    label_seq, length_seq = model.predict(sess, model_restore_path, input_x, args.sigma, actions_dict)
                    
                recognition = []
                for i in range(len(label_seq)):
                    recognition = np.concatenate((recognition, [label_seq[i]]*int(0.5*vid_len*length_seq[i]/args.nRows)))
                recognition = np.concatenate((observed_content,recognition))
                diff = int((0.5+obs_p)*vid_len)-len(recognition)
                for i in range(diff):
                    recognition = np.concatenate((recognition, [label_seq[-1]]))
                #write results to file
                for pred_p in pred_percentages:
                    path=args.results_save_path+"/obs"+str(obs_p)+"-pred"+str(pred_p)
                    write_predictions(path, f_name, recognition[:int((pred_p+obs_p)*vid_len)])
 
