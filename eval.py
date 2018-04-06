#!/usr/bin/python2.7

import numpy as np
import argparse
import glob
import re

def read_sequences(filename, ground_truth_path, obs_percentage):
    # read ground truth
    gt_file = args.ground_truth_path + re.sub('\.recog','.txt',re.sub('.*/','/',filename))
    with open(gt_file, 'r') as f:
        ground_truth = f.read().split('\n')[0:-1]
        f.close()
    # read recognized sequence
    with open(filename, 'r') as f:
        recognized = f.read().split('\n')[1].split()
        f.close()
    
    last_frame = min(len(recognized),len(ground_truth))
    recognized = recognized[int(obs_percentage*len(ground_truth)):last_frame]
    ground_truth = ground_truth[int(obs_percentage*len(ground_truth)):last_frame]
    
    return ground_truth, recognized
    
################################################################## 
parser = argparse.ArgumentParser()
parser.add_argument('--obs_perc')
parser.add_argument('--recog_dir')
parser.add_argument("--mapping_file", default="./data/mapping_bf.txt")
parser.add_argument('--ground_truth_path', default='./data/groundTruth')

args = parser.parse_args()

    
obs_percentage = float(args.obs_perc)
classes_file=open(args.mapping_file,'r')
content = classes_file.read()

classes = content.split('\n')[:-1]
for i in range(len(classes)):
    classes[i]=classes[i].split()[1]
    
filelist = glob.glob(args.recog_dir + '/P*')

n_T=np.zeros(len(classes))
n_F=np.zeros(len(classes))

for filename in filelist:
    gt, recog = read_sequences(filename, args.ground_truth_path, obs_percentage)
    for i in range(len(gt)):
        if gt[i]==recog[i]:
            n_T[classes.index(gt[i])]+=1
        else:
            n_F[classes.index(gt[i])]+=1
##################################################################
acc=0
n=0
for i in range(len(classes)):
    if n_T[i]+n_F[i] !=0:
        acc+=float(n_T[i])/(n_T[i]+n_F[i])
        n+=1
print "MoC  %.4f"%(float(acc)/n)

