# Ground up ANN for MNIST number set

## Initial Setup
In the root directory:<br>
Download Eigen git submodule:<br>
```
git submodule update --init
```
Initial build file configuration:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
```
---

## Build project from Root:
### Release (faster)
```
cmake --build build --config Release
.\build\Release\mnist_ann.exe
```

### Debug (slower)
```
cmake --build build --config Debug
.\build\Debug\mnist_ann.exe
```
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
| 0.005         | 98.24%   | 19     |
