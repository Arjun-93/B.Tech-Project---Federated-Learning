"""_summary_
Author : Arjun Mehra
Dated:  10th November, 2022
In this MLP class we implement the following :

    Returns:        
        ~ __init__
        ~ Weight initialization functions (Zero_init, random_init,....)
        ~ Activation functions
        ~ Derivatives of activation functions
        ~ Forward Propagation
        ~ Backward Prorogation
        ~ Fit
        ~ Predict function
        ~ Predict_proba
        ~ score
    
"""

import numpy as np
import pandas as pd

class MLP():
    '''
    Multi layer Perceptron 
    '''

    def __init__(self, N, A, lr=0.01, epochs=100, batch_size=128, act_func="relu", initialize_weights="zero__init"):
        self.N = N # Number of layers in network
        self.A = A # List of size specifying of neurons in each layer
        self.lr = lr # learning rate
        self.epochs = epochs # Number of Iterations
        self.batch_size = batch_size # Initialize size of Batch
        self.act_func = act_func # Activation Function chosen
        self.initialize_weights = initialize_weights 
        # self.weight_initialization(initialize_weights)
        
        if self.initialize_weights == "zero__init__":
            return self.zero__init__(A)
        
        elif self.initialize_weights == "random__init__":
            return self.random__init__(A)    
        
        elif self.initialize_weights == "random__init__":
            return self.normal__init__(A)
    
    # Zero initialization of weights  
    def zero__init__(self,layers_dims):
        parameters = []
        L = len(layers_dims)
        for l in range(1,L):
            parameters[l] = np.zeros((layers_dims[l], layers_dims[l-1]))
            # parameters['Bias'+str(l)] = np.zeros((layers_dims[l], 1))
        return parameters
    
    # Random initialization of weights
    def random__init__(self,layers_dims):
        np.random.seed(1)
        parameters = {}
        L = len(layers_dims)
        for l in range(1,L):
            parameters['Weights' + str(l)] = np.random.rand((layers_dims[l], layers_dims[l-1]))
            parameters['Bias'+str(l)] = np.zeros((layers_dims[l], 1))
        return parameters

    # Normal initialization of weights
    def normal__init(self,layers_dims):
        np.random.seed(1)
        parameters = {}
        L = len(layers_dims)
        for l in range(1,L):
            parameters['Weights' + str(l)] = np.random.normal(0, 1 , size = (layers_dims[l], layers_dims[l-1]))
            parameters['Bias'+str(l)] = np.zeros((layers_dims[l], 1))
        return parameters
    
    # Activation Functions -->
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def relu(self,z):
        return np.maximum(0, z)

    def tanh(self,z):
        return np.tanh(z)

    def leaky_relu(self,z):
        return np.maximum(0.01*z, z)
    
    def softmax(self,z):
        return np.exp(z) / sum(np.exp(z))


    # Activation Functions derivatives -->
    def sigmoid_derivative(self,z):
        s=1.0 / (1.0+ np.exp(-z))
        return s*(1-s) 
    
    def relu_derivative(self,z):
        if z==0 or z<0 : 
            z=0
        else:
            z=1
        return z

    def tanh_derivative(self,z):
        return (1 - (np.tanh(z)** 2))

    def leaky_relu_derivative(self,z):
        if z==0 or z<0 : 
            z=0.01
        else:
            z=1
        return z
    
    # Forward propagation 
    def forward(self, X):
        input_layer = X
        yA = []
        yZ = []
        for i in range(self.N - 1):
            ni = input_layer.shape[0]
            bi = np.ones([ni, 1])
            yA[i] = input_layer
            yZ[i+1] = np.concatenate(np.matmul(input_layer, self.weights[i].transpose()) +  bi ,axis = 1)
            output_layer = self.act_func(yZ[i+1])
        yA[self.N - 1] = output_layer
        return yA, yZ
    
    # Backward propagation
    def backward(self, X, Y):
        xi = X.shape[0]
        yA, yZ = self.forward(X)
        gradj = []
        gradj[-1] = yA[-1] -Y
        
        for i in np.arange(self.N -2, 0 ,-1):
            theta = self.weights[i]
            theta = np.delete(theta, np.s_[0] - 1)
            gradj[i] = (np.matmul(theta.transpose(), gradj[i+1].transpose() ),np.transpose()* self.act_func(yZ[i]))
        # Compute gradients
        gradients = [None] * (self.N - 1)
        for j in range(self.N - 1):
            grads_tmp = np.matmul(gradj[j + 1].transpose() , yA[j])
            grads_tmp = grads_tmp / xi
            gradients[j] = grads_tmp;
        return gradients
    
    # Fitting the parameters in the model
    def fit(self, X, Y, epochs, reset=False):
        n_examples = Y.shape[0]
        if reset:
            self.initialize_weights()
        for iteration in range(epochs):
            self.gradients = self.backpropagation(X, Y)
            self.gradients_vector = self.unroll_weights(self.gradients)
            self.theta_vector = self.unroll_weights(self.theta_weights)
            self.theta_vector = self.theta_vector - self.gradients_vector
            self.theta_weights = self.update_weights(self.theta_vector)
            
    def unroll_weights(self, rolled_data):
        unrolled_array = np.array([])
        for one_layer in rolled_data:
            unrolled_array = np.concatenate((unrolled_array, one_layer.flatten("F")) )
        return unrolled_array
    
    def update_weights(self, unrolled_data):
        next_layer = self.A.copy()
        next_layer.pop(0)
        updated_weights = []
        for A, next_layer in zip(self.size_layers, next_layer):
            n_weights = (next_layer * (A + 1))
            temp = unrolled_data[0 : n_weights]
            temp = temp.reshape(next_layer, (A+ 1), order = 'F')
            updated_weights.append(temp)
            unrolled_data = np.delete(unrolled_data, np.s_[0:n_weights])
        return updated_weights
    
    def predict(self, X):
        A , Z = self.forward(X)
        Y_hat = A[-1]
        return Y_hat

    def score(predictions, Y):
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size
    
    def predict_proba():    
        pass


        
    
    
    
    