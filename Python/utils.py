#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 19:07:38 2020

@author: siraaj
"""

"""
This library aims to give the user functions built to quickly form/perform: 
    
Neural networks,
regular expressions manipulations, 
uncommon mathematical functions,
logical expressions, 
logical inference, 
graph representations, 
easy graphing
"""

import numpy as np
import tensorflow as tf

###################
# Neural Networks #
################### 

class Layer_Dense:
    
    def __init__(self, inputs, weights, biases):
        self.inputs = inputs
        self.weights = weights
        self.biases = biases
        
    def forward(self):
        layer_outputs = np.dot( np.array(self.inputs), np.array(self.weights).T) + np.array(self.biases)  # Weights first becuse output must be indexed by weights        
        return layer_outputs




class NN:

    def __init__():
        pass









































