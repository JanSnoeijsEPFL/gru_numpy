#translating EDF files from database (full dataset for patient = 1)
#This file reduces the input resolution 
# This combines CNN + LSTM algorithm for EEG sequence classification  
import pyedflib
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
import os
#import pandas as pd
import re
import target_data_gen
from target_data_gen import get_sizes
from target_data_gen import target_gen
from keras.layers import Dense, Activation, LSTM, Dropout, Bidirectional, TimeDistributed, Conv2D, MaxPooling2D, Flatten, GRU

np.random.seed(0)
#defining parameters
output = 1
#X=np.array(sigbufs
batch_size = 100 #(length)
batch_mod = 10000
dataset_size = 18


X, Y = list(), list()




#**********************CALLLING DATA GENERATOR FUNCTION ***********************
X, input_size, length = get_sizes(X, dataset_size)
#******************************************************************************
print(X.shape)
#Defining sizes for input/target data
Y=np.zeros((dataset_size, batch_mod, output))
#yhat=np.zeros((dataset_size, batch_mod, output))


#***********************CALLING TARGET GENERATOR FUNCTION**********************
file = 'database/chb01-summary.txt'
Y = target_gen(output, batch_mod, dataset_size, batch_size, file)
#******************************************************************************
print(Y.shape)



   
# #RESHAPING ACCORDING TO NEW DATASET SIZE AND DATA SET SELECTION
# #DEFINING NEW SHAPES

#inputs
input_size_new = 23
#elements of each frame
batch_size_new = 100
#time steps equivalent 
timesteps= 10 # meaning of this?
#number of sequences
seq_number = 100
#Re-defining dataset size for training
dataset_size = 8


file = 3

#Defining sizes for input/target data
X_new=np.zeros((seq_number*dataset_size, timesteps, input_size_new, batch_size_new, 1))
#yhat=np.zeros((dataset_size, batch_mod, output))
Y_new=np.zeros((seq_number*dataset_size, output))



for m in range(0,dataset_size):
    if (m == 3-1 or  m == 4-1 or m == 15-1 or m == 16-1 or m == 18-1 ):
        if m == 3-1:
            initial_time = 700000
        if m == 4-1:
            initial_time = 300000
        if m == 15-1:
            initial_time = 400000
        if m == 16-1:
            initial_time = 200000
        if m == 18-1:
            initial_time = 400000
    else:
        initial_time = 0
    print('initial time:', initial_time)
    for i in range(0,seq_number):
        for j in range(0,timesteps):
            initial = initial_time+(i*batch_size_new*timesteps)+j*batch_size_new
            final = initial_time+(i*batch_size_new*timesteps)+((j+1)*batch_size_new)
            #print(initial_time+(i*batch_size_new*timesteps)+j*batch_size_new)
            #print(initial_time+(i*batch_size_new*timesteps)+((j+1)*batch_size_new))
            X_new[seq_number*m+i,j,:,0:batch_size_new,0] =  X[m,0,0,0:input_size_new,initial:final,0] 


max_value = np.amax(abs(X_new))
min_value = np.amin(abs(X_new))
# print('MAX VALUE', max_value)
X_new = X_new/max_value # standardization by max value (all values between 0 and 1)

Y_new[seq_number*2+33*2:seq_number*2+39*2,output-1] = 1
Y_new[seq_number*3+37*2:seq_number*3+42*2,output-1] = 1
Y_new[seq_number*14+21*2:seq_number*14+27*2,output-1] = 1
Y_new[seq_number*15+29*2:seq_number*15+37*2,output-1] = 1


print(X_new.shape, Y_new.shape)
# plt.plot(X[2,0,0,0,:,0])
# plt.plot(X[2,0,0,1,:,0])
# plt.show()
# plt.plot(X[3,0,0,0,:,0])
# plt.plot(X[3,0,0,1,:,0])
# plt.show()
#Assigning targets (manually)
#Y[0,2,:] = 1
#Y[3,2,:] = 1


model = Sequential()
# timedistributed conv2D with relu.
# dont understand input shape:
# according to Doc should be: samples, sequence_length, features
# implementation is : None, features, sequence_length, 1
model.add(TimeDistributed(Conv2D(2,(2,2), activation = 'relu'), input_shape=(None,input_size_new,batch_size_new,1)))
# time distributed pooling 2D by maximum.
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
# flattening layer
model.add(TimeDistributed(Flatten()))
# classic GRU, 100 units.
model.add(GRU(100))
# output linear layer with sigmoid as activation.
model.add(Dense(output, activation='sigmoid'))
# loss function is binary cross entropy and optimizer is ADAM
model.compile(loss='binary_crossentropy', optimizer='adam')

print(model.summary())

#training phase, back prop.
model.fit(X_new,Y_new, epochs=40)
#Trying model
yhat = model.predict(X_new, verbose=0)
#weights = model.get_weights() # Getting params
weightsCONV, biasCONV = model.layers[0].get_weights() # Getting params
weightsGRU = model.layers[3].get_weights()
weightsLIN, biasLIN = model.layers[4].get_weights()
file1 = open("CNN_LSTM_dataset_train.txt", 'ab')
np.savetxt(file1, yhat, delimiter="," )
file1.close()
print(np.asarray(weightsLIN).shape, weightsLIN)
fileWEIGHTS = open("KERAS_weights_after_train.txt", 'w')
fileWEIGHTS.write("weights CONV2D\n") 
np.savetxt(fileWEIGHTS, np.asarray(weightsCONV).reshape(4,2), delimiter=",")
fileWEIGHTS.write("bias CONV2D\n") 
np.savetxt(fileWEIGHTS, (np.asarray(biasCONV)), delimiter=", ")
#np.savetxt(fileWEIGHTS, weightsGRU.flatten(), delimiter=",")
fileWEIGHTS.write("weights LIN\n") 
np.savetxt(fileWEIGHTS, np.asarray(weightsLIN), delimiter=", ")
fileWEIGHTS.write("bias LIN\n") 
np.savetxt(fileWEIGHTS, np.asarray(biasLIN), delimiter=", ")
fileWEIGHTS.close()

#########################################################################################################
# #Generating data test 1
file = 18

# #Defining sizes for input/target data
X_test=np.zeros((seq_number, timesteps, input_size_new, batch_size_new, 1))


initial_time = 400000

for i in range(0,seq_number):
    for j in range(0,timesteps):
        initial = initial_time+(i*batch_size_new*timesteps)+j*batch_size_new
        final = initial_time+(i*batch_size_new*timesteps)+((j+1)*batch_size_new)
        #print(initial_time+(i*batch_size_new*timesteps)+j*batch_size_new)
        #print(initial_time+(i*batch_size_new*timesteps)+((j+1)*batch_size_new))
        X_test[i,j,:,0:batch_size_new,0] =  X[file-1,0,0,0:input_size_new,initial:final,0] 
   

   
max_value_2 = np.amax(abs(X_test))
min_value = np.amin(abs(X_test))
# print('MAX VALUE', max_value)
X_test = X_test/max_value_2

# generate predictions, i.e forward pass only with SET weights.
yhat_test = model.predict(X_test, verbose=0)
file2 = open("CNN_GRU_keras_test_18.txt", 'w')
np.savetxt(file2, yhat_test, delimiter="," )
file2.close()

########################################################################################################
#Generating data test 2 this is for non-onset dataset ..
file = 17

# #Defining sizes for input/target data
X_test=np.zeros((seq_number, timesteps, input_size_new, batch_size_new, 1))


initial_time = 4000

for i in range(0,seq_number):
    for j in range(0,timesteps):
        initial = initial_time+(i*batch_size_new*timesteps)+j*batch_size_new
        final = initial_time+(i*batch_size_new*timesteps)+((j+1)*batch_size_new)
        #print(initial_time+(i*batch_size_new*timesteps)+j*batch_size_new)
        #print(initial_time+(i*batch_size_new*timesteps)+((j+1)*batch_size_new))
        X_test[i,j,:,0:batch_size_new,0] =  X[file-1,0,0,0:input_size_new,initial:final,0] 


max_value_2 = np.amax(abs(X_test))
min_value = np.amin(abs(X_test))
X_test = X_test/max_value_2


yhat_test = model.predict(X_test, verbose=0)
file2 = open("CNN_GRU_17_keras.txt", 'w')
np.savetxt(file2, yhat_test, delimiter="," )
file2.close()

