import random
from mlp import BasicFunctions
class Neuron:
    def __init__(self, ninputs, activation=None):
        self.w = [DiffNode(random.uniform(-1,1)) for _ in range(ninputs)]
        self.bias = DiffNode(random.uniform(-1,1))
        self.activation = activation
    
    def __call__(self, inputs):
        assert len(inputs)==len(self.w), "total number of inputs provided and the inputs declared are different"
        act = sum((xi*wi for xi,wi in zip(inputs, self.w)), self.bias)
        return self.activation(act) if self.activation else act
    
    def parameters(self):
        params = self.w+[self.bias]
        params+= self.activation.parameters() if self.activation and hasattr(self.activation, "parameters") else []
        return params

class Layer:
    def __init__(self, ninputs, noutputs, activation):
        self.neurons = [Neuron(ninputs, activation) for _ in range(noutputs)]

    def __call__(self, x):
        out= [neuron(x) for neuron in self.neurons] 
        return out[0] if len(out)==1 else out

    def parameters(self):
        return [params for neuron in self.neurons for params in neuron.parameters()]

class MLP:
    def __init__(self, ninputs, noutputs, activations=None):
        size = [ninputs]+noutputs
        nlayers=len(noutputs)
        if activations is None:
            activations =[Tanh]*nlayers
        elif nlayers!=len(activations):
            raise ValueError("Length of activations must match number of layers.")
        self.layers = [Layer(size[i], size[i+1], activations[i]) for i in range(nlayers)]

    def __call__(self, x):
        for layer in self.layers:
          x= layer(x)
        return x
    
    # resetting gradient to zero for all params
    def zero_grad(self):
        for p in self.parameters():
            p.grad=0

    def parameters(self):
        return [params for layer in self.layers for params in layer.parameters()]
