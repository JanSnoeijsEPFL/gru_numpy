#This file generates automatically input files and sizes
#translating EDF files from database (full dataset for patient = 1)
import pyedflib
import numpy as np
#from matplotlib import pyplot as plt
#import os
#import pandas as pd
import re 

#defining parameters
output = 1
#X=np.array(sigbufs)
batch_size = 10000 #(length)
batch_mod = 100
dataset_size = 7



def get_sizes(X, dataset_size):
    f = pyedflib.EdfReader("database/chb01_{}.edf".format(1))
    #f = pyedflib.EdfReader("database/chb01_01.edf")
    n = f.signals_in_file
    input_size = n
    length = f.getNSamples()[0]
    sigbufs = np.zeros((n, f.getNSamples()[0]))

    for i in np.arange(n):
            sigbufs[i,:] = f.readSignal(i)

    sigbufs_test = sigbufs
    print('SIZE sigbufs', np.shape(sigbufs_test))
    X=np.zeros((dataset_size,1,1, input_size, length,1))
    X[0,0,0,:,:,0]=sigbufs_test 
    for j in range(2,dataset_size+1):
        if (j == 19 or j==28 or j==35 or j==44 or j==45):
            j+=1
        #print(("Reading:dataset_01/chb01_{}.edf ...".format(j)))
        f = pyedflib.EdfReader("database/chb01_{}.edf".format(j))
        for i in np.arange(n):
            sigbufs[i,:] = f.readSignal(i)
        sigbufs_test = sigbufs
        X[j-1,0,0,:,:,0]=sigbufs_test
    #max_value = np.amax(abs(X))
    #X = X/max_value
    print(length)
    return X, input_size, length


#Generating target file for the COMPLETE DATA SET (=7)

def target_gen(output, batch_num, dataset_size, batch_size, file):
    target = np.zeros((dataset_size, batch_num, output))
    filepath = file
    dataset = 0
    with open(filepath) as fp:
        for cnt, line in enumerate(fp): 
            if line.startswith('File Name:'):
                file_num = re.findall(r'\d+',line)
                dataset = int(file_num[1]) 
            if dataset == dataset_size + 1:   #Finish when data set size is achieved
                break                                   
            if line.startswith('Number of Seizures in File: 1'):
                start_time = re.findall(r'\d+',fp.readline())
                end_time = re.findall(r'\d+',fp.readline())
                batch_pos = int(abs((int(start_time[0]))*256/batch_size))
                #print('DATASET', dataset)
                #print('BATCH_POS', batch_pos)
                #print('BATCH_SIZE', batch_size)
                target[dataset-1,batch_pos,:] = 1

    return target 












