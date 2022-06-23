#other option for regularization is dropout
#becoming too dependent on any neuron or for any neuron to be relied upon entirely in a specific instance
'''dropout can help with is co-adoption which happens when neurons depend on the output values of other neurons do not learn the 
underlying function on their own'''
#dropout can also help with noise
#randomly disabling neurons 
#dropout prevent overfitting
#disabling neuron ramdomly via bernoulli distribution

import numpy as np 
import random

dropout_rate = 0.5 

example_output = [0.27, -1.03, 0.99, 0.05, -0.37, -2.01, 1.13, -0.07, 0.73]

while True:
    index = random.randint(0, len(example_output) -1)
    example_output[index] = 0

    dropped_out = 0 
    for value in example_output:
        if value == 0:
            dropped_out += 1
    
    if dropped_out / len(example_output) >= dropout_rate:
        break

print(example_output)

np.random.binomial(2, 0.5, size=10)

dropout_rate = 0.20
np.random.binomial(1, 1-dropout_rate, size=5)

#example_output *= np.random.binomial(1, 1-dropout_rate, example_output.shape)

#dropout is not used while training

class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate
    
    def forward(self, inputs):
        self.inputs = inputs 
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask
    
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

from templatev2 import *
from nnfs.datasets import spiral_data

X,y = spiral_data(samples=1000, classes=3)


dense1 = Layer_Dense(2,512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation1 = Activation_RELU()
dropout1 = Layer_Dropout(0.1)
dense2 = Layer_Dense(512,3)

loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()

optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-5)

for epoch in range(10000):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)

    data_loss = loss_activation.forward(dense2.output, y)

    regularization_loss = loss_activation.loss.regularization_loss(dense1) + \
                          loss_activation.loss.regularization_loss(dense2) 

    loss = data_loss + regularization_loss

    predictions = np.argmax(loss_activation.output, axis=1)

    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'Epoch: {epoch}, ' + f'acc: {accuracy:.3f}, ' +  f'loss: {loss:.3f} ' +  \
                f'data_loss: {data_loss:.3f} ' + f'reg_loss: {regularization_loss:.3f} ' + \
                f'lr: {optimizer.current_learning_rate}')
    
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

X_test, y_test = spiral_data(samples=100, classes=3)

dense1.forward(X_test)
activation1.forward(dense1.output)

dense2.forward(activation1.output)

loss =  loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)

if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_test)

print(f'Validation_acc: {accuracy:.3f}, ' +  f'loss: {loss:.3f} ')


