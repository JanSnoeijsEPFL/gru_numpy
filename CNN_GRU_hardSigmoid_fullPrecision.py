# imports

import numpy as np
from random import randint
import target_data_gen
from target_data_gen import get_sizes
from target_data_gen import target_gen
import model_hard_sigmoid
from model_hard_sigmoid import GRU
from model_hard_sigmoid import Conv2D
from model_hard_sigmoid import MaxPool2D
from model_hard_sigmoid import sigmoid
from model_hard_sigmoid import reLU
from model_hard_sigmoid import tanh
from model_hard_sigmoid import CrossEntropy
from model_hard_sigmoid import Optimizer
from model_hard_sigmoid import orthogonal_initializer

# ***************************************#

print("Initializing ...")
np.random.seed(0)
output = 1
#X=np.array(sigbufs
batch_size = 100 #(length)
batch_mod = 10000
dataset_size = 18
X, Y = list(), list()

#**********************CALLLING DATA GENERATOR FUNCTION ***********************

X, input_size, length = get_sizes(X, dataset_size) #Defining sizes for input/target data
Y=np.zeros((dataset_size, batch_mod, output))

#***********************CALLING TARGET GENERATOR FUNCTION**********************

file = 'database/chb01-summary.txt'
Y = target_gen(output, batch_mod, dataset_size, batch_size, file)

#******************************************************************************

print("Reading edf files ...")

input_size_new = 23
batch_size_new = 100
timesteps= 10 
seq_number = 100 #number of sequences
dataset_size = 18 #Re-defining dataset size for training
#Defining sizes for input/target data
X_new=np.zeros((seq_number*dataset_size, timesteps, input_size_new, batch_size_new, 1))
Y_new=np.zeros((seq_number*dataset_size, output))
loss = 0
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

print( "Normalizing data ...")            
#standardizing data
max_value = np.amax(abs(X_new))
min_value = np.amin(abs(X_new))

X_new = X_new/max_value # standardization by max value (all values between 0 and 1)

print("Setting training labels ...")
#size of data: (800, 10, 23, 100, 1)
Y_new[seq_number*2+33*2:seq_number*2+39*2,output-1] = 1
Y_new[seq_number*3+37*2:seq_number*3+42*2,output-1] = 1
Y_new[seq_number*14+21*2:seq_number*14+27*2,output-1] = 1
Y_new[seq_number*15+29*2:seq_number*15+37*2,output-1] = 1


#***************** TRAINING PHASE ******************************

# sequences, timesteps, features, batches. (100 x nb_files, 10,23, 100,,1)
# define model
# initialize all parameters
sgd_batch_size = int(32)
layer_0 = Conv2D(2,2,2)
# add activation
layer_1 = MaxPool2D()
layer_2 = GRU(sgd_batch_size, timesteps, 1078, 100)
epochs = 50
dataset_size = 8
i = 0
order_SGD = np.arange(0, int(dataset_size*seq_number/sgd_batch_size))

print("Training on ", dataset_size , " files , Epochs = ", epochs, " , SGD batch size = ", sgd_batch_size, ", Optimizer = ADAM ", " , Loss = Binary Cross-entropy\n")
print("Starting training phase ... ")
#forward pass
for n_iters in range(epochs):
    full_loss = 0
    #pick sequence randomly:
    #i = int(randint(200,300)/sgd_batch_size)
    np.random.shuffle(order_SGD)
    for p in range(int(dataset_size*seq_number/sgd_batch_size)):
        i = order_SGD[p]
   
        HConv = layer_0.forward(X_new[i*sgd_batch_size:(i+1)*sgd_batch_size,:,:,:,:]) # 5D data for train_data, 3D for Wconv 2D for Bconv
        YConv = reLU(HConv, deriv=False) # no requirement on shape
        
        # correcting derivative of Max Pool
        DERIV_PROTECT = np.copy(YConv)
        DERIV_PROTECT = DERIV_PROTECT.flatten()
        DERIV_PROTECT[DERIV_PROTECT != 0.] = 1
        DERIV_PROTECT = DERIV_PROTECT.reshape(-1,10,22,99,2)
        
        #pooling layer
        pool_kernel = np.array([2,2])
        YPool, XArgmax = layer_1.forward(YConv,  pool_kernel) #5D data for YConv
        
        #flattening
        X_GRU_flat = YPool.reshape(YPool.shape[0],10,-1) # check size here should be 3D (100*nb_files, 10, 1078)
        
        yhat = layer_2.forward(X_GRU_flat)    #GRU
    
        dy = np.zeros((yhat.shape))
        temp = np.zeros((yhat.shape[0]))
        rep_loss = np.zeros((yhat.shape[0]))
        Ytrue = Y_new[i*sgd_batch_size:(i+1)*sgd_batch_size].reshape(-1)
            
        temp[Ytrue==1] = 2*(yhat-Y_new[i*sgd_batch_size:(i+1)*sgd_batch_size]).reshape(-1)[Ytrue==1]
        temp[Ytrue==0] = 2*(yhat-Y_new[i*sgd_batch_size:(i+1)*sgd_batch_size]).reshape(-1)[Ytrue==0]
        dy=temp.reshape(-1,1)
      
        dsGRU, dxGRU = layer_2.backward(dy, X_GRU_flat)
        dyMaxPool = dxGRU.reshape(dxGRU.shape[0], dxGRU.shape[1],11,49,2)

        dxMaxPool = layer_1.backward(XArgmax, dyMaxPool)

        dxMaxPool_augmented = np.zeros((dxMaxPool.shape[0], dxMaxPool.shape[1], dxMaxPool.shape[2], dxMaxPool.shape[3]+1, dxMaxPool.shape[4]))
        dxMaxPool_augmented[:,:,:,0:dxMaxPool.shape[3],:]=dxMaxPool*DERIV_PROTECT[:,:,:,0:dxMaxPool.shape[3],:]
        dhConv2D = reLU(HConv, deriv=True)*dxMaxPool_augmented
        
        layer_0.backward(dhConv2D, X_new[i*sgd_batch_size:(i+1)*sgd_batch_size,:,:,:,:])
    
        loss, __ = CrossEntropy(yhat, Y_new[i*sgd_batch_size:(i+1)*sgd_batch_size]) # works only for y = 0 or 1
        full_loss += loss
   
    print("iter", n_iters, "---------", "loss =", full_loss)
    
#*******************************TEST**************************#

print("Preparing for test phase ...")
startfile = 17
nb_files = 1
seq_number=100
#Defining sizes for input/target data
X_test=np.zeros((seq_number*nb_files, timesteps, input_size_new, batch_size_new, 1))

layer_2.change_input_size(seq_number*nb_files,10,100)
X_GRU_flat_augmented_test = np.ones((seq_number*nb_files,10,1079))
S_GRU_augmented_test = np.ones((seq_number*nb_files,101))

print("Reading files ", startfile+1, " to ", nb_files + startfile, "\n")
for m in range(startfile,nb_files+startfile):
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
            X_test[seq_number*m+i-startfile*seq_number,j,:,0:batch_size_new,0] =  X[m,0,0,0:input_size_new,initial:final,0]    

print("Normalizing data ... ")  
max_value_2 = np.amax(abs(X_test))
min_value = np.amin(abs(X_test))
# print('MAX VALUE', max_value)
X_test = X_test/max_value_2

print("Generating predictions ...")
HConv = layer_0.forward(X_test) # 5D data for train_data, 3D for Wconv 2D for Bconv
YConv = reLU(HConv, deriv=False) # no requirement on shape
#pooling layer
pool_kernel = np.array([2,2])
YPool, XArgmax = layer_1.forward(YConv,  pool_kernel) #5D data for YConv
#flattening
X_GRU_flat = YPool.reshape(YPool.shape[0],10,-1) # check size here should be 3D (100*nb_files, 10, 1078)
#print(X_GRU_flat.shape)
#GRU
yhat_test = layer_2.forward(X_GRU_flat) # timesteps

file2 = open("results/testfile18_hardSigmoid_lowPrecision_8training.txt", 'w')
np.savetxt(file2, yhat_test, delimiter="," )
file2.close()

print("Predictions saved to file : ", file2)