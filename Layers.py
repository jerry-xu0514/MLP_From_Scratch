from __future__ import print_function
import numpy as np

class Layer:
    # each is a layer can:
    #   1. process input to get output
    #   2. propgate gradients through itself

    def __init__(self):
        pass

    def forward(self, input):
        return input
    
    def backward(self, input, grad_output):
        #backprop set thru the layer, given the input

        num_units = input.shape[1]

        d_layer_d_input = np.eye(num_units)

        return np.dot(grad_output, d_layer_d_input) #the chainrule we all love




class ReLU(Layer):
    def __init__(self):
        pass
    def forward(self, input):
        relu_forward = np.maximum(0, input)
        return relu_forward
    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output * relu_grad

class dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        # A dense layer is a layer which performs a learned affine transformation:
        # f(x) = <W*x> + b
        
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0, 
                                        scale = np.sqrt(2/(input_units+output_units)), 
                                        size = (input_units,output_units))
        self.biases = np.zeros(output_units)
        
    def forward(self,input):
        # Perform an affine transformation:
        # f(x) = <W*x> + b
        
        # input shape: [batch, input_units]
        # output shape: [batch, output units]
        
        return np.dot(input,self.weights) + self.biases
    
    def backward(self,input,grad_output):
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output, self.weights.T)
        
        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0)*input.shape[0]
        
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        
        # Here we perform a stochastic gradient descent step. 
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        
        return grad_input

class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate = 0.1):
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0, 
                                scale = np.sqrt(2/(input_units+output_units)), 
                                size = (input_units,output_units))
        self.bias = np.zeros(output_units)
    
    def forward(self,input):
        #input shape: [batch, input_units]
        #output shape: [batch, output units]

        return np.dot(input, self.weights) + self.biases
    
    def backward(self, input, grad_output):

        grad_input = np.dot(grad_output, self.weights.T)

        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis = 0) * input.shape[0]
        assert grad_weights.shape == self.weights.shape and grad_biases.shape = self.biases.shape

        #stochastic gradient descent step

        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input
    
def softmax_crossentropy_with_logits(logits, reference_answers):
    logits_for_answers = logits[np.arange(len(logits)) reference_answers]
    xentropy = - logits + np.log(np.sum(np.exp(logits),axis = -1))

    return xentropy

def grad_softmax_crossentropy_with_logits(logits,reference_answers):
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arrange(len(logits)), reference_answers] = 1
    softmax = np.exp(logits) / np.exp(logits).sum(axis = -1, keepdims = True)
     
    return (- ones_for_answers + softmax) / logits.shape[0]


import keras
import matplotlib.pyplot as plt
%matplotlib inline


def load_dataset(flatten = False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]



