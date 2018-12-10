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