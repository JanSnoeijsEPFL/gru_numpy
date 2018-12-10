import framework
import torch
import math
import matplotlib.pyplot as plt 
import numpy as np # used only for plotting
# generates the dataset

# area interval : [0, 1]²
# disk radius : 1/sqrt(2*pi)
# samples 1000
X, T=framework.generate_data(1000,0,1, 1/(2*math.pi))
X_test, T_test=framework.generate_data(1000,0,1, 1/(2*math.pi))

#****************************************************************************#

# defines and creates the model

minibatch_size = 100 # batch size for SGD
nb_epochs = 500 # number of epochs for training
batch_mod = int(X.size(0)/minibatch_size) # calculate number of batches
X_mini = torch.Tensor(minibatch_size, batch_mod, X.size(1)) # reorganize dataset: [batch_size X nb_batches X nb_features]
T_mini = torch.Tensor(minibatch_size, batch_mod) # reorganize targets: [batch_size x nb_batches]
for i in range (0, batch_mod): # fill reorganized tensors with data.
    X_mini[:, i, :] = X[i*minibatch_size:((i+1)*minibatch_size),:]
    T_mini[:, i] = T[i*minibatch_size:((i+1)*minibatch_size)]

Model = framework.Sequential(3) # creates the model with 3 layers
layer_0 = Model.add_layer('Tanh', 2, 25) # creates Linear() layer with reLU activation (2 inputs, 25 outputs)
layer_1 = Model.add_layer('ReLU', 25, 25) # creates Linear() layer with Sigmoid activation (25 inputs, 25 outputs)
layer_2 = Model.add_layer('Sigmoid', 25, 2) # creates Linear() layer with Sigmoid activation (25 inputs, 2 outputs)

#***************************************************************************#

# training the network
Yaxis = np.empty(shape =(nb_epochs, 2))
for j in range (0, nb_epochs): # loop over nb_epochs
    '''yplot = Model.forward(X, layer_0, layer_1, layer_2) #only used for plots
    #MSEplot,__, Yplot = Model.MSE(yplot, T)
    #nb_errorsplot = framework.calc_nb_errors(Yplot, T)

    #Yaxis[j, 0] = MSEplot
    #Yaxis[j, 1] = nb_errorsplot'''

    for i in range (0, batch_mod): # loop over the batches 


        yauto = Model.forward(X_mini[:,i,:], layer_0, layer_1, layer_2) # forward pass
        MSE, gradMSE, Yhat = Model.MSE(yauto, T_mini[:,i]) # loss calculation
        Model.backward(gradMSE, layer_0, layer_1, layer_2) # backward pass


yfinal = Model.forward(X, layer_0, layer_1, layer_2) # full train dataset forward pass (with trained parameters)
MSEtrain, grad, Ytrain = Model.MSE(yfinal, T) # calculate MSE error on training dataset
nb_errorstrain = framework.calc_nb_errors(Ytrain, T) # calculate absolute training error
print('MSEtrainError', MSEtrain, '%\nTrainError',  nb_errorstrain, '%\n')



#***************************************************************************#

# testing the network

ytest = Model.forward(X_test, layer_0, layer_1, layer_2) # forward pass on test dataset
MSEtest, gradtest, Ytest = Model.MSE(ytest, T_test) # MSE calculation
nb_errorstest = framework.calc_nb_errors(Ytest, T_test) # absolute error calculation
print('MSEtestError', MSEtest, '%\nTestError',  nb_errorstest, '%\n')  

#****************************************************************************#

#plotting absolute error during training
'''    
Xaxis = np.arange(nb_epochs)
fig = plt.figure()
ax = plt.axes()
plt.plot(Xaxis[1:nb_epochs-1], Yaxis[1:nb_epochs-1,1])
plt.ylabel('absolute error %')
plt.xlabel('epochs')
plt.show()    '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    