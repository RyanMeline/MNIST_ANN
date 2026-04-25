# Ground up ANN for MNIST number set

## Initial Setup
In the root directory:<br>
Download Eigen git submodule:<br>
```
git submodule update --init
```
Initial build file configuration:
```
cmake -S . -B build
```
---

## Build project:
From the root directory, run:
```
cmake --build build
```
Or from /build/ run:
```
cmake --build .
```

The binary will be specified in the output message of CMake<br>
Default for me was /build/Debug/

---

## Results
2 Hidden Layers:
```
    network.add_layer(784, 128, relu, relu_back);
    network.add_layer(128, 64, relu, relu_back);
    network.add_layer(64, 10, softmax);
```
| learning_rate | Accuracy | Epochs |
|---------------|----------|--------|
| 0.001         | 97.88%   | 20     |

---

3 Hidden Layers:
```
    network.add_layer(784, 256, relu, relu_back);
    network.add_layer(256, 128, relu, relu_back);
    network.add_layer(128, 64, relu, relu_back);
    network.add_layer(64, 10, softmax);
```
| learning_rate | Accuracy | Epochs |
|---------------|----------|--------|
| 0.001         | 97.96%   | 17     |
