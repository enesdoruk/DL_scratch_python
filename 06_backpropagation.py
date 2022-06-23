from turtle import forward
import numpy as np 

x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0

#forward prop
xw0 = x[0]*w[0]
xw1 = x[1]*w[1]
xw2 = x[2]*w[2]

#fin calculate
z = xw0 + xw1 + xw2 + b

#relu
y = max(z,0)

#derivative from the next layer
dvalue= 1.0

relu_dz = dvalue * (1 if z > 0 else 0)

#partil derivative of the multiplication, the chain rule
dsum_dxw0 = 1
drelu_dxw0 = relu_dz * dsum_dxw0

dsum_dxw1 = 1
drelu_dxw1 = relu_dz * dsum_dxw1

dsum_dxw2 = 1
drelu_dxw2 = relu_dz * dsum_dxw2

dsum_db = 1
drelu_db = relu_dz * dsum_db


dmu1_dx0 = w[0]
drelu_dx0 = drelu_dxw0 * dmu1_dx0

dmu1_dx1 = w[1]
drelu_dx1 = drelu_dxw1 * dmu1_dx1

dmu1_dx2 = w[2]
drelu_dx2 = drelu_dxw2 * dmu1_dx2

dmu1_dw0 = x[0]
drelu_dw0 = drelu_dxw0 * dmu1_dw0

dmu1_dw1 = x[1]
drelu_dw1 = drelu_dxw1 * dmu1_dw1

dmu1_dw2 = x[2]
drelu_dw2 = drelu_dxw2 * dmu1_dw2

dx = [drelu_dx0, drelu_dx1, drelu_dx2] #gradinets of inputs
dw = [drelu_dw0, drelu_dw1, drelu_dw2] #gradients of weights
db = drelu_db

w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
b += -0.001 * db
#####################

dvalues = np.array([[1., 1., 1.],[2., 2., 2.], [3., 3., 3.]])
weights = np.array([[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]).T
inputs = np.array([[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]])
z = np.array([[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]])
biases = np.array([[2, 3, 0.5]])

dinputs =  np.dot(dvalues, weights.T)
dweights = np.dot(inputs.T, dvalues)
dbiases = np.sum(dvalues, axis=0, keepdims=True)

#keepsdim lets us keep the gradient as a row vector

dvalues = np.array([[1., 1., 1., 1.],[2., 2., 2., 2.], [3., 3., 3., 3.]])

drelu = np.zeros_like(z)
drelu[z > 0] = 1
drelu *= dvalues
#####################

dvalues = np.array([[1., 1., 1.],[2., 2., 2.], [3., 3., 3.]])
weights = np.array([[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]).T
inputs = np.array([[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]])
biases = np.array([[2, 3, 0.5]])

layer_output = np.dot(inputs, weights) + biases
relu_output = np.maximum(0, layer_output)

drelu = relu_output.copy()
drelu[layer_output <= 0] = 0

dinputs = np.dot(drelu, weights.T)
dweights = np.dot(inputs.T, drelu)
dbiases = np.sum(drelu, axis=0, keepdims=True)

weights += -0.001 * dweights
biases += -0.001 * dbiases
#####################

class Layer_Dense:
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_RELU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
#####################

class Loss:
    def calculate(self, output, y):
        sample_loss = self.forward(output, y)
        data_loss = np.mean(sample_loss)

        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1: #shape == 1 is list otherwise array
            y_true = np.eye(labels)[y_true] #list to one hot encoded vectors
            #np.eye create nxn diagonal matrix
        
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs /samples #gradient normalization

#optimizer sum all of the gradients related to each weight and bias before multipliyng them by the learning rate.
#gradient normalization for low equation cost
#####################

softmax_output = [0.7, 0.1, 0.2]
softmax_output = np.array(softmax_output).reshape(-1,1)

softmax_eye = np.eye(softmax_output.shape[0])
softmax_eye = softmax_output * np.eye(softmax_output.shape[0]) # == np.diagflat(softmax_output)

class Activation_Softmax:
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues) #create uninitialized array

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Activation_Softmax_Loss_CategoricalCrossEntropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()
    
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

#####################
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_RELU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_val / np.sum(exp_val, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues) #create uninitialized array

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

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

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1: #shape == 1 is list otherwise array
            y_true = np.eye(labels)[y_true] #list to one hot encoded vectors
            #np.eye create nxn diagonal matrix
        
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs /samples #gradient normalization

class Activation_Softmax_Loss_CategoricalCrossEntropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()
    
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


import nnfs
from nnfs.datasets import spiral_data
nnfs.init()


X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_RELU()

dense2 = Layer_Dense(3,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y)
print("loss = ", loss)

loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)
