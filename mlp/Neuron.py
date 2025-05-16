import random
from mlp import BasicFunctions
class Neuron:
    def __init__(self, ninputs):
        self.w = [DiffNode(random.uniform(-1,1)) for _ in range(ninputs)]
        self.bias = DiffNode(random.uniform(-1,1))
    
    def __call__(self, inputs):
        assert len(inputs)==len(self.w), "total number of inputs provided and the inputs declared are different"
        return sum((xi*wi for xi,wi in zip(inputs, self.w)), self.bias)
    
    def parameters(self):
        return self.w+[self.bias]

class Layer:
    def __init__(self, ninputs, noutputs, activation):
        self.neurons = [Neuron(ninputs) for _ in range(noutputs)]
        self.activation = activation

    def __call__(self, x):
        out= [self.activation(neuron(x)) for neuron in self.neurons] 
        return out[0] if len(out)==1 else out

    def parameters(self):
        return [params for neuron in self.neurons for params in neuron.parameters()]

class MLP:
    def __init__(self, ninputs, noutputs, activation=None):
        size = [ninputs]+noutputs
        nlayers=len(noutputs)
        if activations is None:
            activations =[tanh]*nlayers
        elif nlayers!=len(activations):
            raise ValueError("Length of activations must match number of layers.")
        self.layers = [Layer(size[i], size[i+1], activation[i]) for i in range(nlayers)]

    def __call__(self, x):
        for layer in self.layers:
          x= layer(x)
        return x

    def parameters(self):
        return [params for layer in self.layers for params in layer.parameters()]
