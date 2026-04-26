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
| 0.008         | 98.21%   | 17     |

---

### Shuffling training data
3 Hidden Layers:
```
std::vector<int> index(train_data.images.size());
std::iota(index.begin(), index.end(), 0); //fills indecies

//Called each Epoch
std::shuffle(index.begin(), index.end(), std::mt19937{std::random_device{}()});
```
| learning_rate | Accuracy | Epochs |
|---------------|----------|--------|
| 0.008         | 98.32%   | 19     |
| 0.005         | 98.31%   | 17     |
| 0.005         | 98.42%   | 29     |
| 0.01          | 98.31%   | 19     |
| 0.001         | 98.19%   | 17     |

---

### Learning Rate Scheduler
With Data Shuffling and 3 Hidden Layers<br>
Step Decay:
```
if(i % 5 == 0) learning_rate *= decay_rate;
```
| learning_rate | decay rate |Accuracy | Epochs |
|---------------|------------|---------|--------|
| 0.01          | 0.1        | 98.42%  | 13     |
| 0.01          | 0.5        | 98.44%  | 10     |
| 0.01          | 0.5        | 98.46%  | 15     |
| 0.01          | 0.25       | 98.56%  | 11     |
| 0.01          | 0.75       | 98.57%  | 18     |
| 0.01          | 0.75       | 98.67%  | 15     |