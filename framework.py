# includes
import torch
import math

#**************************************************************************************************#

# prepares data inputs and target tensors
# returns data inputs (normalized) and target values (1 inside the circle, 0 outside)
# sizes inputs: [nb_samples x 2] targets: [nb_samples]
# generic number of samples, area interval and disk radius
def generate_data(nb_points, lower_lim, upper_lim, disc_radius2):
    inp = torch.Tensor(nb_points, 2).uniform_(lower_lim, upper_lim)
    targets = torch.le(torch.sum(inp.pow_(exponent=2), dim=1), disc_radius2)
    mean = inp.mean()   # normalization of inputs
    std_dev = inp.std()
    inputs = (inp - mean)/std_dev
    return inputs, targets.type(torch.Tensor)


#**************************************************************************************************#

# implements a fully connected layer: 'Y=sum(W*X)+B'
class Linear():
    epsilon = 1e-2     # variance for weight initialization
    x = torch.Tensor() # internal parameter (input vector received when calling "Linear_object.forward()"
    
    # initialization function
    # should be called only once per instance of the class
    # insize is the number of input features or the number of neurons in the previous layer
    # outsize is the number of neurons of the layer
    def __init__(self, insize, outsize): 
        self.insize =  insize # gets input and output sizes
        self.outsize = outsize
        self.w = torch.Tensor(insize, outsize).normal_(0, self.epsilon) # initializes the parameters.
        self.b = torch.Tensor(1, outsize).normal_(0, self.epsilon)
        self.lr = 0.1 # learning rate
       
    # forward pass calculation
    # receives input tensor of shape : [nb_samples x insize]
    def forward(self, x): 
        # [nb_samples x insize] x [insize x outsize] + [nb_samples x outsize] => [nb_samples x outsize]
        output = torch.matmul(x, self.w)+self.b
        self.x = x # stores x in class parameter to reuse for backward()
        return output
    
    # backward pass calculation
    # receives gradient of the next layer
    # returns gradient with respect to the input. Updates weights and biases.
    def backward(self, gradwrt_output):
        # matrix multiplication [nb_samples, outsize]*[insize, outsize]^t => [nb_samples x insize]
        # sum over samples for parameters (normal Gradient Descent. If divided into batches -> SGD)
        gradwrt_input = torch.matmul(gradwrt_output,torch.t(self.w))
        self.w -= self.lr*torch.matmul(torch.t(self.x),gradwrt_output) # update weights
        self.b -= self.lr*torch.sum(gradwrt_output, 0) # update biases
        return gradwrt_input
    
    # returns outputsize and weights
    def param(self):
        return self.outsize, self.w


#**************************************************************************************************#    

# implements Rectifed linear unit in forward and backward pass
class ReLU():
    # internal parameter received during forward pass (reuse for backward)
    x = torch.Tensor()
    
    # calculates an element-wise reLU() on X: [nb_samples x insize] and returns it
    def forward(self, x):
        output = torch.max(x, torch.Tensor(x.size(0), x.size(1)).fill_(0)) # fill with 0 if lower than 0
        self.x = x # store x in class parameter
        return output
    
    # calculates an element-wise derivative reLU() on X and returns the elementwise product with gradwrt_output
    # X and gradwrt_output have automatically the same size because forward () doesn't change the size of the tensor.
    def backward(self, gradwrt_output):
        p = torch.max(self.x, torch.Tensor(self.x.size(0), self.x.size(1)).fill_(0)) # fill with 0 if lower than 0 
        q = torch.ceil(p) # round the values to upper integer (all the values strictly above 0 have derivative = 1)
        deriv_relu = torch.min(q, torch.Tensor(self.x.size(0), self.x.size(1)).fill_(1)) # fill with 1 if above 1
        gradwrt_input = deriv_relu*gradwrt_output # element-wise multiplication
        return gradwrt_input

#*************************************************************************************************#

# implements sigmoid activation function
class Sigmoid():
    # internal parameter (output of foward pass received during forward() call)
    y = torch.Tensor()
    
    # forward pass. receives X: [nb_samples x insize]
    # returns sigmoid(X)
    def forward(self, x):
        self.y = 1/(1+torch.exp(-x)) # element-wise calculation and storage into class parameter
        return self.y
    
    # backward pass. recevies gradient of next layer (of same size as input tensor of forward())
    def backward(self, gradwrt_output):
        deriv_sigmoid = self.y*(1-self.y) # calculate derivative with respect to the output
                                          # avoids more exponential-type computations.
        gradwrt_input = deriv_sigmoid*gradwrt_output # element_wise product
        return gradwrt_input

#*************************************************************************************************#

# implements tanh activation function
class Tanh():
    # internal parameter (output of foward pass received during forward() call)
    y = torch.Tensor()
    
    # forward pass. receives X: [nb_samples x insize]
    # returns tanh(X)
    def forward(self, x):
        self.y = 1-(2/(torch.exp(2*x)+1)) # element-wise calculation and storage into class parameter
        return self.y
    
    # backward pass. recevies gradient of next layer (of same size as input tensor of forward())
    def backward(self, gradwrt_output):
        deriv_tanh = 1-torch.pow(self.y,2) # element_wise, defined w.r.t output of forward pass
                                           # avoids more exponential-type computations
        gradwrt_input = deriv_tanh*gradwrt_output # element-wise products
        return gradwrt_input

#*************************************************************************************************#

# creates model, trains the model, calculates error, tests the model (on user-command)
class Sequential():
    
    # initialization: sets number of layers (reused in forward and backward methods)
    def __init__(self, nb_layers):
        self.nb_layers = nb_layers
    
    # creates a new layer
    # returns class objects of layer and activation respectively
    def add_layer(self,activation, insize, outsize):
        new_layer = Linear(insize, outsize) # initializes a linear layer
        
        if activation == 'Sigmoid': # initializes an activation layer based on parameter.

            new_layer_activation = Sigmoid()

        elif activation == 'ReLU':

            new_layer_activation  = ReLU()

        elif activation == 'Tanh':

            new_layer_activation  = Tanh()

        return new_layer, new_layer_activation

    # does the forward pass on the complete model 
    # argument 1: X[nb_samples x nb_features] network input data
    # argument 2 : tuple of class objects. index 0 should be Linear(), index 1 should be an activation layer.
    def forward(self, X, *layer): 
        vars()['layer_act0'] = X  # dynamic variable
        for i in range(1, self.nb_layers+1): # loop over the number of layers in the model
            vars()['layer'+str(i)] = layer[i-1][0].forward(vars()['layer_act'+str(i-1)]) # forward pass of Linear layer
            vars()['layer_act'+str(i)] = layer[i-1][1].forward(vars()['layer'+str(i)]) # forward pass of activation layer
            
        return vars()['layer_act'+str(self.nb_layers)] # return last output of forward pass
    
    # backpropagation on the complete model.
    # argument 1: gradient of loss function with respect to network's output [nb_samples x model_outputsize]
    # argument 2: tuple of class objects.index 0 should be Linear(), index 1 should be an activation layer.
    # doesnt return anything, just updates all parameters. (weights and biases)
    def backward(self, gradwrtoutput, *layer): 
        vars()['layer_grad'+str(self.nb_layers)] = gradwrtoutput # create new dynamic variables to store the gradients
        for i in range(0, self.nb_layers): # loop over the number of layers
            # backward pass on activation layer (starting from the output layer)
            vars()['layer_grad_act'+str(self.nb_layers-i-1)] = \
                layer[self.nb_layers-i-1][1].backward(vars()['layer_grad'+str(self.nb_layers-i)])
            # forward pass on Linear layer ()
            vars()['layer_grad'+str(self.nb_layers-i-1)] =  \
                layer[self.nb_layers-i-1][0].backward(vars()['layer_grad_act'+str(self.nb_layers-i-1)])
    
    # calculates MSE loss
    # Yhat: tensor size of [nb_samples]x[2]: output of network
    # T  : tensor size of [nb_samples] : target vector
    # returns the MSE error, the gradient of the loss function w.r.t. the output layer, and the vector of classified values
    def MSE(self, Yhat, T):
        __, Y = torch.max(Yhat, 1)  # argmax function
        errorMSE = torch.sum(1/2*(Y.type(torch.Tensor) - T).pow(2))*100/T.size(0) # MSE loss in percentage
        GradMSE = Yhat - T.reshape(-1,1) # Values of output layer minus target: same shape as the output layer.
        for i in range(0, Y.size(0)): # fill gradient vector with 0 for the neurons which didnt provide the maximum value.
            if Y[i] == 0:
                GradMSE[i][1]=0
            else:
                GradMSE[i][0]=0
        return  float(errorMSE), GradMSE, Y.type(torch.Tensor)
    
#***************************************************************************************************#

# calculate absolute errors (to be used after training and after testing)
def calc_nb_errors(y, T):
    nb_errors = torch.sum(abs(y-T))*100/T.size(0)
    return float(nb_errors)