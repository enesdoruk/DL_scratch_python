#two types activation func = for hidden layer and output layer
#generally activation func used for hidden layers will be the same
#step func = greater than 0 is 1, lowet than zeros is 0 

#for a NN to fit a nonlinear function, we need it to contain two or more hidden layers and we need to use nonlinear activation func

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

for i in inputs:
    if i > 0:
        output.append(i)
    else:
        output.append(0)
print(output)    

for i in inputs:
    output.append(max(0,i))

from asyncio.constants import ACCEPT_RETRY_DELAY
import numpy as np
output = np.maximum(0, inputs)

#activation func should be inside forward func

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_RELU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


from nnfs.datasets import spiral_data

X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2,3)
activation1 = Activation_RELU()

dense1.forward(X)
activation1.forward(dense1.output)

print(activation1.output[:5])
#####################

#softmax func on the output data can take in non-normalized, or uncalibrated, inputs and produce a normalized distribution.

layer_output = [4.8, 1.21, 2.238]
E = 2.71

exp_val = []
for output in layer_output:
    exp_val.append(E ** output)

#exp for non negative val

norm_base = sum(exp_val)
norm_val = []
for value in exp_val:
    norm_val.append(value / norm_base)

exp_val = np.exp(inputs)
probabilities = exp_val / np.sum(exp_val, axis=1, keepdims=True)

#axis = 0 => row, axis = 1 => column

class Activation_Softmax:
    def forward(self, inputs):
        exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_val / np.sum(exp_val, axis=1, keepdims=True)

        self.output = probabilities
    
softmax = Activation_Softmax()
softmax.forward([[1,2,3]])
print(softmax.output)


dense1 = Layer_Dense(2,3)
activation1 = Activation_RELU()
dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output)

class Activation_RELU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)