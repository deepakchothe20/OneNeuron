# OneNeuron
One Neuron | Perceptron

``` python
def __init__(self, eta, epochs):
    self.weights = np.random.randn(3) * 1e-4 # SMALL WEIGHT INIT
    logging.info(f"initial weights before training: \n{self.weights}")
    self.eta = eta # LEARNING RATE
    self.epochs = epochs
```

## Table
x1| x2|x3
-|-|-
0|0|1
0|1|1

## Pointer
* Point 1
* point 2
1 Point 1
2 Point 2