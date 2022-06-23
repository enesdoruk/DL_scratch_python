#####################
#inputs size and weights size should be same

inputs = [1.0, 2.0, 3.0]
weights = [0.2, 0.43, 0.12]
bias = 5.0

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
#####################

inputs = [1.0, 2.0, 3.0]

weights1 = [0.3, 0.6, 0.12]
weights2 = [0.35, 0.632, 0.52]
weights3 = [-0.3, -0.6, -0.22]

bias1 = 2
bias2 = 5
bias3 = 0.5

output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + bias1, #neuron1
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + bias2, #neuron2
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + bias3] #neuron3

#each neuron is connected the same input. but weights and bias is different.
#seperate weights and bias that each neuron applies to the input this is called fully connected.
#every neuron is the current layer has connections to every neuron from the previous layer.
#####################

inputs = [1.0, 2.0, 3.0]

weights = [[0.3, 0.6, 0.12],
            [0.35, 0.632, 0.52],
            [-0.3, -0.6, -0.22]]

biases = [2, 6, 12]

layer_output = []

for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    
    neuron_output += neuron_bias
    layer_output.append(neuron_output)

print(layer_output)

#zip func lets us iterate multiple iterables
#####################

#tensor are closely related to arrays
#homologous = [[3,1], [3,1]] unhomologous = [[2,1], [3,1]]
# matrix is rectangular array and has rows and columns
#(3,2,4) = 3 matrix, each matrix has two rows and each rows has 4 elements => 3D array
#A tensor object is an object that can be represented as an array
#linear array one dimensional array
#elementwise multip via dot product. dot products are used for vectors. both vectors must be same size

dot_product = inputs[0]*biases[0] + inputs[1]*biases[1] + inputs[2]*biases[2]

#we need multiply weights and inputs of the same index
#####################

import numpy as np
inputs = [1.0, 4.0, 6.0, 23.5, 12.4]
weights = [1.0, 0.20, -2.0, 0.5, 0.4]
bias = 3.0 

output = np.dot(inputs, weights) + bias

from numba import jit
import time 
import warnings
warnings.filterwarnings("ignore")

start = time.time()

@jit( cache=True)
def num(inputs, weights, bias):
    output = np.dot(inputs, weights) + bias
    return output

numb_output = num(inputs, weights, bias)
stop = time.time()
print("time = ", stop - start, " output = ", numb_output)
#####################

inputs = [1.0, 2.0, 3.0]

weights = [[0.3, 0.6, 0.12],
            [0.35, 0.632, 0.52],
            [-0.3, -0.6, -0.22]]

biases = [2, 6, 12]

layer_output = np.dot(weights, inputs) + biases
#####################

#neural networks tend to receive data in batches
#NN expect to take many samples for faster to train in batches, batches help generalization.
#matrix product is an op which have 2 matrices and perform dot products of all combinations of rows and columns. 
#for matrix product s must be (a,b) (b,c) shape
# a= [1,2,4] => row vector 
#we use transpose for matrix multiplication

a = [1,2,3]
np.array(a) == np.expand_dims(np.array(a), axis=0)

#expand dims adds a new dimension at the index of the axis

at = np.array(a).T

#dot product and matrix product is the same func in numpy = np.dot
#####################

inputs = [[1.3, 2.6, 0.12, 2.0],
            [3.35, 1.632, 0.52, 3.0],
            [5.3, 1.6, 2.22, 1.0]]


weights = [[0.3, 0.6, 0.12, 2.0],
            [0.35, 0.632, 0.52, 1.0],
            [-0.3, -0.6, -0.22, 0.2]]

biases = [2, 6, 12, 4]

output = np.dot(inputs, np.array(weights).T) + biases