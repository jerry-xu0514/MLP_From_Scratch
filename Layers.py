from __future__ import print_function
from turtle import forward
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