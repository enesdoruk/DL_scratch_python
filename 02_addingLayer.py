#NN become deep when they have 2 or more hidden layers

inputs = [[1.3, 2.6, 0.12, 2.0],
            [3.35, 1.632, 0.52, 3.0],
            [5.3, 1.6, 2.22, 1.0]]


weights = [[0.3, 0.6, 0.12, 2.0],
            [0.35, 0.632, 0.52, 1.0],
            [-0.3, -0.6, -0.22, 0.2]]

biases = [2, 6, 12]

weights2 = [[0.3, 0.6, 0.12],
            [0.35, 0.632, 0.52],
            [-0.3, -0.6, -0.22]]

biases2 = [2, 6, 12]

weights3 = [[0.3, 0.6, 0.12, 0.12 ],
            [0.35, 0.632, 0.52, 0.12],
            [-0.3, -0.6, -0.22, 0.12]]

biases3 = [2, 6, 12, 21]

import numpy as np

# 4 features 2 hidden layers 3 of 3 neurons each 
layer1_output = np.dot(inputs, np.array(weights).T) + biases
layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2
layer3_output = np.dot(layer2_output, np.array(weights3)) + biases3
#####################

from nnfs.datasets import spiral_data
import nnfs
import matplotlib.pyplot as plt

nnfs.init()

X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:,0], X[:,1])
plt.show()

plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
plt.show()
#####################

#dense == fully connecte layer 
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #init weights and bias
        pass
    def forward(self, inputs):
        #calculate output
        pass

#weights are often initialized randomly for a model without pretraned weights

def __init__(self, n_inputs, n_neurons):
    self.weights = 0.01*np.random.randn(n_inputs, n_neurons) # instead of transpose use this queue old = (neurons, inputs)
    self.biases = np.zeros((1, n_neurons))

#output of the neuron is 0 this means dead neurons. After neurons take the output
#gaussuian distributions with mean of 0 and variance of 1. center is 0, range is -1 and +1

np.random.randn(2,5)
np.zeros((2,5)) 

n_inputs = 2
n_neurons = 4

weights = 0.01 * np.random.randn(n_inputs, n_neurons)
biases = np.zeros((1, n_neurons))

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

dense1 = Layer_Dense(2,3)
dense1.forward(X)

print(dense1.output[:5])

#there 5 rows and 3 columns. 3 value is 3 neurons in dense1.