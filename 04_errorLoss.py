#loss function == cost function
#accuracy is simply applying argmax to the output to find the index of the biggest value. 
#The output of NN as actually confidence.
#squared error or mean squared error for regresion
#categorical cross entropy for classification and generally use with softmax func
#log loss for binary logistic regression
#cross entropy compares two probability distributions

import math
from typing_extensions import dataclass_transform

from torch import negative 

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = - (math.log(softmax_output[0])*target_output[0] +
          math.log(softmax_output[1])*target_output[1] +
          math.log(softmax_output[2])*target_output[2])

print(loss)

#categorical cross entropy loss accounts for that and outputs a larger loss the lower confidence
#confidence level 1 == %100 sure about prediction

import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]])
class_target = np.array([[1,0,0], [0,1,0], [0,1,0]])

if len(class_target.shape) == 1:
    correct_confidence = softmax_outputs[range(len(softmax_outputs)) ,class_target]
elif len(class_target.shape) == 2:
    correct_confidence = np.sum(softmax_outputs * class_target, axis=1)

neg_log = -np.log(correct_confidence)
average_loss = np.mean(neg_log)
#####################

class Loss:
    def calculate(self, output, y):
        sample_loss = self.forward(output, y)
        data_loss = np.mean(sample_loss)

        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1- 1e-7) #prevent division by 0

        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples) ,y_true]

        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihood = -np.log(correct_confidence)

        return negative_log_likelihood

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(softmax_outputs, class_target)
print(loss)
#####################

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_RELU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_val / np.sum(exp_val, axis=1, keepdims=True)

        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_loss = self.forward(output, y)
        data_loss = np.mean(sample_loss)

        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1- 1e-7) #prevent division by 0

        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples) ,y_true]

        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihood = -np.log(correct_confidence)

        return negative_log_likelihood

from nnfs.datasets import spiral_data

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_RELU()
dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossEntropy()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss = loss_function.calculate(activation2.output, y)
print("loss = ", loss)
#####################

softmax_outputs = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]])
class_target = np.array([[0,1,1]])

predictions = np.argmax(softmax_outputs, axis=1)
if len(class_target.shape) == 2:
    class_target = np.argmax(class_target, axis=1)
accuracy = np.mean(predictions == class_target)
print("acc = ", accuracy)