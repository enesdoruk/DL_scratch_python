from random import sample
from cv2 import _OutputArray_DEPTH_MASK_8S
from templatev2 import *
from nnfs.datasets import spiral_data
import numpy as np

class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs 
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

#calculation error in regression is mean squared error and mean absolute error

class Loss_MeanSquredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples

#mean absolute error used as loss, penalizes the error linearly. MAE (L1) loss used less frequently than MSE (L2)

class Loss_MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        outputs = len(dvalues[0])

        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples

'''
With a regression model, we have two problems. first problem is that each output neuron in the model is a seperate output like 
in a binary regression model and unlike a classifier where all outputs contribute toward a common prediction.
second problem is that the prediction is a float values, and we cant simply check if the output value equals the ground truth one
'''

#there is no perfect way to show accuracy. true = 14.500 pred = 14.450. is so close but not true.
# for this reason generaly use precision

'''
accuracy_precision = np.std(y)
predicitons = activation2.output
accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)
'''

from nnfs.datasets import sine_data

X, y = sine_data()

dense1 = Layer_Dense(1,64)
activation1 = Activation_RELU()
dense2 = Layer_Dense(64,64)
activation2 = Activation_RELU()
dense3 = Layer_Dense(64,1)
activation3 = Activation_Linear()

loss_function = Loss_MeanSquredError()

optimizer = Optimizer_Adam(learning_rate=0.005, decay=1e-3)

accuracy_precision = np.std(y) / 250 

for epoch in range(10000):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    data_loss = loss_function.calculate(activation3.output, y)

    regularization_loss = loss_function.regularization_loss(dense1) + \
                          loss_function.regularization_loss(dense2) + \
                          loss_function.regularization_loss(dense3)

    loss = data_loss + regularization_loss

    predictions = activation3.output 
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)

    if not epoch % 100:
        print(f'Epoch: {epoch}, ' + f'acc: {accuracy:.3f}, ' +  f'loss: {loss:.3f} ' +  \
                f'data_loss: {data_loss:.3f} ' + f'reg_loss: {regularization_loss:.3f} ' + \
                f'lr: {optimizer.current_learning_rate}')
    
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

import matplotlib.pyplot as plt 

X_test, y_test  = sine_data()

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

plt.plot(X_test, y_test)
plt.plot(X_test, activation3.output)
plt.show()