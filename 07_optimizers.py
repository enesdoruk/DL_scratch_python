#stochastic gradient descent = fits a single sample at a time 
#vanilla gradient descent, gradient descent, batch gradient descent = fit whole dataset at once
#mini batch gradient descent = fit slices of a dataset 

from audioop import bias
from lib2to3.pgen2.token import OP
from template import * 

class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
    
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases


import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,64) #2 input features 64 output values
activation1 = Activation_RELU()

dense2 = Layer_Dense(64,3) #64 input features 3 output values
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()

optimizer = Optimizer_SGD()

for epoch in range(10000):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y ,axis=1)

    accuracy = np.mean(predictions == y)

    if not epoch % 10:
        print(f'Epoch: {epoch},' + f'acc: {accuracy:.3f}, ' +  f'loss: {loss:.3f}')

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)


start_learning_rate = 1.
learning_rate_decay = 0.1

for step in range(20):
    learning_rate = start_learning_rate * (1. / (1 + learning_rate_decay * step))
    print(learning_rate)


class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0 

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1 + self.decay * self.iterations))

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases
    
    def post_update_params(self):
        self.iterations += 1
#####################

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,64) #2 input features 64 output values
activation1 = Activation_RELU()

dense2 = Layer_Dense(64,3) #64 input features 3 output values
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()

optimizer = Optimizer_SGD(decay=1e-2)

for epoch in range(10000):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y ,axis=1)

    accuracy = np.mean(predictions == y)

    if not epoch % 10:
        print(f'Epoch: {epoch},' + f'acc: {accuracy:.3f}, ' +  f'loss: {loss:.3f} ' +  f'lr: {optimizer.current_learning_rate}')

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
#####################

class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0 
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1 + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:
            weight_updates = - self.current_learning_rate * layer.dweights
            bias_updates = - self.current_learning_rate * layer.diases

        layer.weights += weight_updates
        layer.biases += bias_updates
    
    def post_update_params(self):
        self.iterations += 1

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,64) #2 input features 64 output values
activation1 = Activation_RELU()

dense2 = Layer_Dense(64,3) #64 input features 3 output values
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()

optimizer = Optimizer_SGD(decay=1e-3, momentum=0.5 )

for epoch in range(10000):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y ,axis=1)

    accuracy = np.mean(predictions == y)

    if not epoch % 10:
        print(f'Epoch: {epoch},' + f'acc: {accuracy:.3f}, ' +  f'loss: {loss:.3f} ' +  f'lr: {optimizer.current_learning_rate}')

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

class Optimizer_Adad:
    def __init__(self, learning_rate=1, decay=0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay 
        self.iterations = 0
        self.epsilon = epsilon 
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1 + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Optimizer_RMSProp:
    def __init__(self, learning_rate=1, decay=0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay 
        self.iterations = 0
        self.epsilon = epsilon 
        self.rho = rho
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1 + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1-self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1-self.rho) * layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adam:
    def __init__(self, learning_rate=1, decay=0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay 
        self.iterations = 0
        self.epsilon = epsilon 
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1 + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1-self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1-self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1-self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1-self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_momentums_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_momentums_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

# prevent overfitting is to change the model size 
# model is not learning at all, try a larger model
# model learning but there is a divergence between training and test data, try smaller model
# regularization and dropout other techniques.

#cross validation is good for small dataset
#cross val = ex split data 3 cunk. traing first two, last valid. first valid, others train. second valid, otherst train.