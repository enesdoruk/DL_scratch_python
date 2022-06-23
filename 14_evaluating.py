from xml.sax.saxutils import prepare_input_source
import numpy as np
from zmq import XPUB_MANUAL_LAST_VALUE
from templatev3 import *

class Model:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)
    
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)

        self.trainable_layers = []

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

            self.loss.remember_trainable_layers(self.trainable_layers)
        
        if isinstance(self.layers[-1], Activation_Softmax) and \
            isinstance(self.loss, Loss_CategoricalCrossEntropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossEntropy()

    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        self.accuracy.init(y)

        train_steps = 1

        if validation_data is not None:
            validation_steps = 1

            X_val, y_val = validation_data

            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(validation_steps):
                if batch_size is None:
                    batch_x = X_val
                    batch_y = y_val
                else:
                    batch_x = X_val[step*batch_size:(step+1)*batch_size]
                    batch_y = y_val[step*batch_size:(step+1)*batch_size]

                output = self.forward(batch_x, training=False)
                self.loss.calculate(output, batch_y)

                predictions = self.output_layer_activation.predictions(output)
                self.accuracy.calculate(predictions, batch_y)
            
            validation_loss = self.loss.calculate_accumulated()
            validation_accuracy = self.accuracy.calculate_accumulated()

            print(f'validation, ' + f'acc: {validation_accuracy:.3f} ' + 
                    f'loss: {validation_loss:.3f}')
       
        if batch_size is not None:
            train_steps = len(X) // batch_size

            if train_steps * batch_size < len(X):
                train_steps += 1
            
            if validation_data is not None:
                validation_steps = len(X_val) // batch_size

                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        for epoch in range(1, epochs+1):
            print(f'epoch: {epoch}')

            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_x = X
                    batch_y = y
                else:
                    batch_x = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                output = self.forward(batch_x, training=True)

                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not step % print_every or step == train_steps -1:
                    print(f'Step: {step}, ' + f'acc: {accuracy:.3f}, ' +  f'loss: {loss:.3f} ' +  \
                        f'data_loss: {data_loss:.3f} ' + f'reg_loss: {regularization_loss:.3f} ' + \
                        f'lr: {self.optimizer.current_learning_rate}')

            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' + f'acc: {epoch_accuracy:.3f}, ' +  f'loss: {epoch_loss:.3f} ' +  \
                        f'data_loss: {epoch_data_loss:.3f} ' + f'reg_loss: {epoch_regularization_loss:.3f} ' + \
                        f'lr: {self.optimizer.current_learning_rate}')


            if validation_data is not None:
                self.loss.new_pass()
                self.accuracy.new_pass()

                for step in range(validation_steps):
                    if batch_size is None:
                        batch_x = X_val
                        batch_y = y_val
                    else:
                        batch_x = X_val[step*batch_size:(step+1)*batch_size]
                        batch_y = y_val[step*batch_size:(step+1)*batch_size]

                    output = self.forward(batch_x, training=False)

                    self.loss.calculate(output, batch_y)

                    predictions = self.output_layer_activation.predictions(output)

                    self.accuracy.calculate(predictions, batch_y)

                validation_loss = self.loss.calculate_accumulated()
                validation_accuracy = self.accuracy.calculate_accumulated()

                print(f'Validation, ' + f'acc: {validation_accuracy:.3f}, ' +  f'loss: {validation_loss:.3f}')

    def evaluate(self, X_val, y_val, *, batch_size=None):
        validation_steps = 1

        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
        
        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
            if batch_size is None:
                batch_x = X_val
                batch_y = y_val
            else:
                batch_x = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]

            output = self.forward(batch_x, training=False)

            self.loss.calculate(output, batch_y)

            predictions = self.output_layer_activation.predictions(output)

            self.accuracy.calculate(predictions, batch_y)

        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f'Validation, ' + f'acc: {validation_accuracy:.3f}, ' +  f'loss: {validation_loss:.3f}')

    def forward(self, X, training):
        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        
        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)

            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            
            return 
        
        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    

x, y, x_test, y_test = create_data_mnist('fashion_mnist_images')

keys = np.array(range(x.shape[0]))
np.random.shuffle(keys)
x = x[keys]
y = y[keys]

x = (x.reshape(x.shape[0], -1).astype(np.float32) - 127.5) / 127.5 # -1,1
x_test = (x_test.reshape(x_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5 # -1,1

model = Model()
model.add(Layer_Dense(x.shape[1], 128))
model.add(Activation_RELU())
model.add(Layer_Dense(128, 128))
model.add(Activation_RELU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

model.set(loss=Loss_CategoricalCrossEntropy(),
            optimizer=Optimizer_Adam(decay=1e-4),
            accuracy=Accuracy_Categorical())

model.finalize()

model.train(x, y, validation_data=(x_test, y_test), epochs=10, batch_size=128, print_every=100)

model.evaluate(x_test, y_test)