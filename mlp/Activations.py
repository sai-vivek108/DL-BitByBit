# Activation function encapsulated as callable objects
    # hyperboloc tangent activation funtion
class Tanh:
    def __call__(self, x):
        t= (math.exp(2*x.data)-1)/(math.exp(2*x.data)+1)
        out=DiffNode(t, (x,))

        def _backward():
            x.grad+=(1-t**2)*out.grad
        out._backward = _backward
        return out
    
    def parameters(self):
        return []
    
    # sigmoid function for probabilities
class Sigmoid:
    def __call__(self, x):
        s = 1/(1+math.exp(-x.data))
        out = DiffNode(s, (x,))

        def _backward():
            x.grad+= s*(1-s)     # first derivative: sigmoid(x) * (1- sigmoid(x))
        out._backward = _backward
        return out

    def parameters(self):
        return []

    # Parametric ReLu
class PReLu:
    def __init__(self, alpha=0.01):
        self.alpha = alpha if isinstance(alpha, DiffNode) else DiffNode(alpha)   

    def __call__(self,x):
        out = DiffNode(self.alpha.data*x.data if x.data<0 else x.data, (x,self.alpha))
        def _backward():
            x.grad+=(self.alpha.data if x.data<0 else 1) * out.grad
            self.alpha.grad+=(x.data if x.data<0 else 0)* out.grad
        out._backward = _backward
        return out
    
    def parameters(self):
        return [self.alpha]
