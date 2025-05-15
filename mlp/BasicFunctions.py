class DiffNode:
    """To process a single scalar value"""
    def __init__(self, input, _children=()):
        self.data = input
        self._backward = lambda: None
        self.grad = 0
        self._prev = set(_children)     # set of parent nodes

    def __repr__(self):
        return f"data = {self.data}, grad = {self.grad}"#, parent = {self._prev}

    #divison: self/other (but we substituted it for multiplication)
    def __truediv__(self, other):
        other = other if isinstance(other, DiffNode) else DiffNode(other)
        out = DiffNode(self.data/other.data, (self, other))
        def _backward():
            self.grad+= (other.data)**-1 * out.grad  #d/ds(s/o)
            other.grad+=(-self.data) * (other.data)**-2 * out.grad  #d/do(s/o)
        out._backward = _backward
        return out

    # multiplication: self * other
    def __mul__(self, other):
        # if other is not the instance of same class
        other = other if isinstance(other, DiffNode) else DiffNode(other)
        out = DiffNode(self.data * other.data, (self, other))
        def _backward():
            self.grad+= other.data*out.grad
            other.grad+= self.data*out.grad
        out._backward = _backward
        return out
    
    # if multiplied in reverse order, eg: other* data
    def __rmul__(self, other):
        return self*other

    # addition: self + other
    def __add__(self, other):
        # instance check
        other = other if isinstance(other, DiffNode) else DiffNode(other)
        out = DiffNode(self.data + other.data, (self, other))
        def _backward():
            self.grad+= 1 * out.grad
            other.grad+= 1 * out.grad
        out._backward = _backward
        return out
    
    # if added in reverse order: other + self
    def __radd__(self, other):
        return self+other
    
    # subtraction: self - other
    def __sub__(self, other):        
        other = other if isinstance(other, DiffNode) else DiffNode(other)
        out = DiffNode(self.data - other.data, (self, other))
        def _backward():
            # negative due to the differentiation of self
            self.grad+= -other.data*out.grad
            other.grad+= self.data*out.grad
        out._backward = _backward
        return out
    
    # if subtracted in reverse order: other - self
    def __rsub__(self, other):
        return self-other

    # power: self^other
    def __pow__(self,other):
        if isinstance(other, (int, float)):
            out = DiffNode(self.data**other, (self, other))
            def _backward():
                self.grad+= (other) * (self.data)**(other-1)
            out.grad = _backward
        elif isinstance(other, DiffNode):
            out= DiffNode(self**other.data, (self, other))
            def _backward():
                self.grad+= (other.data) * (self.data)**(other.data-1)
            out._backward = _backward
        else:
            raise TypeError(f"Unsupported type: {type(other)}. This function currently supports only int/float/DiffNode instance")
        
        return out
    
    # exponential
    def exp(self):
        out= DiffNode(math.exp(self.data), (self))
        def _backward():
            self.grad+=out.data*out.grad
        return out
    
    # hyperboloc tangent activation funtion
    def tanh(self):
        x=self.data
        t= (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out=DiffNode(t, (self,))

        def _backward():
            self.grad+=(1-t**2)*out.grad
        out._backward = _backward
        return out
    
    # back propagation (using toplogical sort)
    # The reason to choose Topological sort is because it ensures that no node is visited before all it's 
    # dependencies are processed. Without it, when we call _gradient, it wpuld use the gradient that hasn't been updated
    def backward(self):
        visited = set()
        topo = []        # list to store the neurons in order
        def build_topo(node):
            if node not in visited: # check if node is already visited
                visited.add(node)
                for child in node._prev:
                    build_topo(child)   # recursively going through the child nodes until we reach the parent
                topo.append(node)   # post-order once the parent is found

        
        build_topo(self)
        # print(topo)
        self.grad=1
        for node in topo[::-1]:
            # print("node.data: ", node.data)
            node._backward()    # go in reverse topological order
            # print("grad: ", node.grad)
# x = DiffNode(2.0)
# y = x * x + x
# y.backward()