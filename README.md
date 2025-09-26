# NanoGrad

NanoGrad is a lightweight automatic differentiation engine written from scratch in Python. It helps you calculate gradients of any expression efficiently, making it perfect for building and training neural networks.

## Features

- Automatic differentiation with reverse-mode autodiff
- Dynamic computational graphs
- Support for basic mathematical operations
- Neural network building blocks
- Simple and readable implementation

## Installation

```bash
git clone https://github.com/ItsRikan/NanoGrad.git
cd NanoGrad
pip install -r requirements.txt
```

## Project Structure

```
NanoGrad/
├── layers/                      
│   ├── __init__.py
│   ├── activation.py            # contains all types of activation function (e.g, Tanh,ReLU,Softmax)
│   ├── linear.py                # contains linear layers (e.g, Linear,BatchNorm1D)
├── Optimizers/                 
│   ├── __init__.py
│   ├── adagrad.py               # contains adagrad implementation            
│   ├── adam.py                  # contains adam implementation
│   └── sgd.py                   # contains sgd implementation
├── util/      
│   ├── __init__.py
│   ├── to_matrics.py            # contains matrics checker and converter
│   ├── util_for_file.py         # importing and saving with json and pickle support
│   └── util.py                  # contains diension alignment function
├── testing/
│   ├── test.py
│   └── testing.ipynb  
├── .gitignore
├── engine.py                    # contains implementation of Matrics
├── functional.py                # contains MAE, MSE, Binary Cross Entropy loss functions
├── LICENSE                      
├── model.py                     # contains Sequential
├── nn.py                        # imported linear + optimizers
├── README.md
└── requirements.txt
```

## Quick Start

```python
from nanograd.engine import Matrics

# Create some Matrics
x = Matrics([[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19]])
y = Matrics([[20,21,22,23,24],[25,26,27,28,29],[30,31,32,33,34],[35,36,37,38,39]])

# Perform operations
z = x * y + Matrics([[1.0]])
z.backward()

# Access gradients
print(x.grad)  # dy/dx
print(y.grad)  # dy/dy
```

## Neural Network Example

```python
from nanograd.nn import Linear, Tanh
import numpy as np

# Create a simple neural network
model = Sequential([
    Linear(2, 4),
    Tanh(),
    Linear(4, 1)
])

# Train the model
X = np.random.randn(100, 2)
y = np.random.randn(100, 1)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- Built for educational purposes to understand automatic differentiation

## Contact

[Rikan Maji] - [@Its_Py_Coder](https://twitter.com/Its_Py_Coder)
Project Link: [https://github.com/ItsRikan/NanoGrad](https://github.com/ItsRikan/NanoGrad.git)
This project is licensed under the MIT License - see the LICENSE file for details.
